# ComfyUI-QwenVL-Utils
# HuggingFace model backend for QwenVL
# Optimized for maximum inference speed

import gc
import os
from typing import Optional, Tuple, Any, Dict

import numpy as np
import torch
from PIL import Image

# Performance: Set environment variables before importing transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() // 2)))

try:
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        AutoConfig,
        BitsAndBytesConfig,
    )
    # Try to import Qwen2VLForConditionalGeneration first (more compatible)
    # Falls back to AutoModelForVision2Seq if not available
    try:
        from transformers import Qwen2VLForConditionalGeneration
        _MODEL_CLASS = Qwen2VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq
            _MODEL_CLASS = AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForCausalLM
            _MODEL_CLASS = AutoModelForCausalLM
except ImportError as e:
    error_msg = (
        "\n" + "="*70 + "\n"
        "[QwenVL-Utils] ERROR: transformers library not found or outdated\n"
        "="*70 + "\n"
        f"Import error: {e}\n\n"
        "Installation required:\n"
        "  pip install transformers>=4.37.0\n\n"
        "For quantization support (optional):\n"
        "  pip install bitsandbytes>=0.41.0\n\n"
        "For better performance (optional):\n"
        "  pip install accelerate\n"
        "="*70
    )
    raise ImportError(error_msg) from e

from .config import HF_ALL_MODELS, Quantization, SYSTEM_PROMPTS
from .attention import resolve_attention_mode, SageAttentionContext
from .utils import (
    get_device_info,
    normalize_device_choice,
    clear_memory,
    enforce_memory_limits,
    ensure_hf_model,
    tensor_to_pil,
    sample_video_frames,
    get_optimal_dtype,
    optimize_model_for_inference,
    try_torch_compile,
    is_bf16_supported,
)


# Performance: Check for static cache support (transformers 4.38+)
def _static_cache_available() -> bool:
    """Check if static KV cache is available for faster generation"""
    try:
        from transformers import StaticCache  # noqa: F401
        return True
    except ImportError:
        return False


# Performance: Check for CUDA Graphs support
def _cuda_graphs_available() -> bool:
    """Check if CUDA graphs can be used for faster inference"""
    if not torch.cuda.is_available():
        return False
    try:
        # CUDA Graphs require compute capability >= 7.0
        major, _ = torch.cuda.get_device_capability()
        return major >= 7
    except Exception:
        return False


def get_quantization_config(model_name: str, quantization: Quantization) -> Tuple[Optional[BitsAndBytesConfig], Optional[torch.dtype]]:
    """Get quantization configuration for model loading with optimal dtype selection"""
    info = HF_ALL_MODELS.get(model_name, {})
    
    # Pre-quantized models don't need additional quantization
    if info.get("quantized"):
        return None, None
    
    if quantization == Quantization.Q4:
        try:
            # Performance: Use BF16 compute dtype if available (faster on Ampere+)
            compute_dtype = torch.bfloat16 if is_bf16_supported() else torch.float16
            cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            return cfg, None
        except Exception as e:
            error_msg = (
                f"\n[QwenVL-Utils] ERROR: Failed to configure 4-bit quantization: {e}\n"
                "This usually means bitsandbytes is not installed or incompatible.\n\n"
                "Install bitsandbytes:\n"
                "  pip install bitsandbytes>=0.41.0\n\n"
                "Note: Requires CUDA-capable GPU. For CPU inference, use FP16 or GGUF models.\n"
            )
            raise RuntimeError(error_msg) from e
    
    if quantization == Quantization.Q8:
        try:
            return BitsAndBytesConfig(load_in_8bit=True), None
        except Exception as e:
            error_msg = (
                f"\n[QwenVL-Utils] ERROR: Failed to configure 8-bit quantization: {e}\n"
                "Install bitsandbytes: pip install bitsandbytes>=0.41.0\n"
            )
            raise RuntimeError(error_msg) from e
    
    # FP16/BF16 - use optimal dtype for hardware
    dtype = get_optimal_dtype()
    return None, dtype


class HFModelBackend:
    """HuggingFace model backend for QwenVL inference - Optimized for speed"""
    
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        self._sage_attention_enabled = False
        self._use_static_cache = _static_cache_available()
        self._warmup_done = False
        print(f"[QwenVL-Utils] HF Backend initialized on {self.device_info['device_type']}")
        if self._use_static_cache:
            print("[QwenVL-Utils] Static KV cache available (faster generation)")
    
    def clear(self):
        """Clear all loaded models and free memory"""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        self._sage_attention_enabled = False
        self._warmup_done = False
        clear_memory()
    
    def load_model(
        self,
        model_name: str,
        quant_value: str,
        attention_mode: str,
        use_compile: bool,
        device_choice: str,
        keep_model_loaded: bool,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        """Load or reuse a model with the specified configuration"""
        # Parse quantization
        quantization = enforce_memory_limits(
            model_name, 
            Quantization.from_value(quant_value), 
            self.device_info
        )
        
        # Resolve attention mode
        attn_impl, use_sage = resolve_attention_mode(attention_mode)
        self._sage_attention_enabled = use_sage
        if use_sage:
            print(f"[QwenVL-Utils] Attention backend: {attn_impl} + SageAttention (experimental)")
        else:
            print(f"[QwenVL-Utils] Attention backend: {attn_impl}")
        
        # Resolve device
        device_requested = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        device = normalize_device_choice(device_requested)
        
        # Create signature for caching
        signature = (model_name, quantization.value, attn_impl, device, use_compile, min_pixels, max_pixels)
        
        # Check if model is already loaded with same config
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            return
        
        # Clear existing model
        self.clear()
        
        # Download/locate model
        model_path = ensure_hf_model(model_name)
        
        # Get quantization config
        quant_config, dtype = get_quantization_config(model_name, quantization)
        
        # Prepare loading kwargs with performance optimizations
        load_kwargs = {
            "device_map": device if device != "auto" else "auto",
            "attn_implementation": attn_impl,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,  # Memory optimization
            "trust_remote_code": True,  # Required for Qwen models
            "ignore_mismatched_sizes": True,  # Handle minor architecture variations
        }
        
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        
        # Performance: Enable memory-efficient attention if available
        if attn_impl in ("flash_attention_2", "sdpa"):
            # These attention implementations are memory efficient by default
            pass
        
        print(f"[QwenVL-Utils] Loading {model_name} ({quantization.value}, attn={attn_impl}, dtype={dtype})")
        
        # Load model using the appropriate model class with error handling
        try:
            # Load model - let transformers handle config automatically
            # Don't pass explicit config to avoid compatibility issues between Qwen2-VL and Qwen3-VL
            self.model = _MODEL_CLASS.from_pretrained(model_path, **load_kwargs)
            print(f"[QwenVL-Utils] Model loaded successfully: {type(self.model).__name__}")
        except (RuntimeError, AttributeError) as e:
            error_str = str(e)
            if "size mismatch" in error_str or "has no attribute" in error_str:
                print(f"[QwenVL-Utils] WARNING: Compatibility issue detected. Attempting fallback loading...")
                # Try loading without strict checking and with force_download
                load_kwargs_relaxed = load_kwargs.copy()
                load_kwargs_relaxed["ignore_mismatched_sizes"] = True
                # Remove potentially problematic attention implementation
                if "attn_implementation" in load_kwargs_relaxed:
                    original_attn = load_kwargs_relaxed["attn_implementation"]
                    load_kwargs_relaxed["attn_implementation"] = "eager"
                    print(f"[QwenVL-Utils] Switching attention from {original_attn} to eager for compatibility")
                
                try:
                    self.model = _MODEL_CLASS.from_pretrained(model_path, **load_kwargs_relaxed)
                    print("[QwenVL-Utils] Model loaded successfully with fallback settings")
                except Exception as inner_e:
                    error_msg = (
                        f"\n{'='*70}\n"
                        f"[QwenVL-Utils] ERROR: Failed to load model {model_name}\n"
                        f"{'='*70}\n"
                        f"Original error: {e}\n"
                        f"Retry error: {inner_e}\n\n"
                        f"This usually means:\n"
                        f"1. Model requires a newer version of transformers\n"
                        f"2. Model files are corrupted or incomplete\n"
                        f"3. Qwen3-VL requires transformers >= 4.46.0\n\n"
                        f"Try:\n"
                        f"1. Update transformers: pip install --upgrade transformers>=4.46.0\n"
                        f"2. Delete model cache: {model_path}\n"
                        f"3. Check model compatibility: https://huggingface.co/{model_name}\n"
                        f"{'='*70}"
                    )
                    raise RuntimeError(error_msg) from inner_e
            else:
                raise
        
        self.model = optimize_model_for_inference(self.model)
        
        # Performance: Enable gradient checkpointing for memory efficiency during inference
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        
        # Apply torch.compile if requested
        if use_compile:
            self.model = try_torch_compile(self.model, device)
        
        # Load processor with pixel settings
        processor_kwargs = {"trust_remote_code": True}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        
        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.current_signature = signature
    
    @torch.inference_mode()
    def generate(
        self,
        prompt_text: str,
        image: Optional[torch.Tensor],
        video: Optional[torch.Tensor],
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        num_beams: int,
        repetition_penalty: float,
    ) -> str:
        """Generate text from the model - Optimized for speed"""
        # Build conversation
        conversation = [{"role": "user", "content": []}]
        
        # Add image if provided
        if image is not None:
            pil_image = tensor_to_pil(image)
            if pil_image is not None:
                conversation[0]["content"].append({"type": "image", "image": pil_image})
        
        # Add video frames if provided
        if video is not None:
            frames = [tensor_to_pil(frame) for frame in sample_video_frames(video, frame_count)]
            frames = [f for f in frames if f is not None]
            if frames:
                conversation[0]["content"].append({"type": "video", "video": frames})
        
        # Add text prompt
        conversation[0]["content"].append({"type": "text", "text": prompt_text})
        
        # Process conversation directly with the processor
        # This is the recommended approach for Qwen2-VL and Qwen3-VL
        # The processor handles chat template, tokenization, and image/video encoding together
        text_prompt = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Get images and videos from conversation
        image_inputs = []
        video_inputs = []
        
        for item in conversation[0]["content"]:
            if item["type"] == "image":
                image_inputs.append(item["image"])
            elif item["type"] == "video":
                video_inputs.extend(item["video"])
        
        # Process inputs using the correct method for multimodal input
        # For Qwen3-VL: processor expects text and media together
        processor_kwargs = {
            "text": [text_prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        
        # Add images if present
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        
        # Add videos if present (as a list of frame lists)
        if video_inputs:
            processor_kwargs["videos"] = [video_inputs]
        
        try:
            processed = self.processor(**processor_kwargs)
        except Exception as e:
            error_msg = (
                f"[QwenVL-Utils] Failed to process inputs: {e}\n"
                f"Text: {len(text_prompt)} chars\n"
                f"Images: {len(image_inputs)}\n"
                f"Video frames: {len(video_inputs)}"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Move to model device with optimized transfer
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # Performance: Move tensors efficiently and ensure contiguity
        model_inputs = {}
        for key, value in processed.items():
            if torch.is_tensor(value):
                # Ensure tensor is contiguous for optimal memory access
                value = value.contiguous()
                # Move to device (non-blocking for CUDA)
                if model_device.type == "cuda":
                    value = value.to(model_device, non_blocking=True)
                else:
                    value = value.to(model_device)
            model_inputs[key] = value
        
        # Performance: Synchronize to ensure data is ready
        if model_device.type == "cuda":
            torch.cuda.current_stream().synchronize()
        
        # Prepare generation kwargs
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)
        
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,  # Performance: Always use KV cache
        }
        
        # Configure sampling
        if num_beams == 1:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        else:
            gen_kwargs["do_sample"] = False
        
        # Performance: Use contrastive search for potentially faster, better quality generation
        # (Only when not using beam search and temperature is low)
        if num_beams == 1 and temperature < 0.3:
            gen_kwargs.update({
                "penalty_alpha": 0.6,
                "top_k": 4,
            })
        
        # Generate with optional SageAttention
        with SageAttentionContext(enable=self._sage_attention_enabled):
            outputs = self.model.generate(**model_inputs, **gen_kwargs)
        
        # Sync and decode
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        input_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        
        return text.strip()
    
    def run(
        self,
        model_name: str,
        quantization: str,
        preset_prompt: str,
        custom_prompt: str,
        image: Optional[torch.Tensor],
        video: Optional[torch.Tensor],
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        num_beams: int,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        attention_mode: str,
        use_torch_compile: bool,
        device: str,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Tuple[str]:
        """Run inference with the model"""
        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Resolve prompt
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        
        # Load model
        self.load_model(
            model_name=model_name,
            quant_value=quantization,
            attention_mode=attention_mode,
            use_compile=use_torch_compile,
            device_choice=device,
            keep_model_loaded=keep_model_loaded,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        
        try:
            # Generate
            text = self.generate(
                prompt_text=prompt,
                image=image,
                video=video,
                frame_count=frame_count,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
            )
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


# Global instance for reuse across calls
_hf_backend_instance = None


def get_hf_backend() -> HFModelBackend:
    """Get or create the HF backend instance"""
    global _hf_backend_instance
    if _hf_backend_instance is None:
        _hf_backend_instance = HFModelBackend()
    return _hf_backend_instance
