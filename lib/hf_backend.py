# ComfyUI-QwenVL-Utils / lib / hf_backend.py
# HuggingFace model backend for QwenVL inference

import os
import subprocess
from typing import Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm

try:
    import comfy.utils
    import comfy.model_management
    _COMFY = True
except ImportError:
    _COMFY = False

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Model class detection
_QWEN3_CLS = _QWEN2_5_CLS = _QWEN2_CLS = _FALLBACK_CLS = None
try:
    from transformers import Qwen3VLForConditionalGeneration as _QWEN3_CLS  # type: ignore
except ImportError:
    pass
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _QWEN2_5_CLS  # type: ignore
except ImportError:
    pass
try:
    from transformers import Qwen2VLForConditionalGeneration as _QWEN2_CLS  # type: ignore
except ImportError:
    pass
try:
    from transformers import AutoModelForVision2Seq as _FALLBACK_CLS  # type: ignore
except ImportError:
    from transformers import AutoModelForCausalLM as _FALLBACK_CLS  # type: ignore

from .settings import HF_ALL_MODELS, Quantization, SYSTEM_PROMPTS
from .attention import resolve_attention_mode, SageAttentionContext
from .device import (
    get_device_info, normalize_device_choice, clear_memory,
    enforce_memory_limits, is_bf16_supported, get_optimal_dtype,
)
from .model_utils import ensure_hf_model, optimize_model_for_inference, try_torch_compile
from .media import tensor_to_pil, sample_video_frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_class(model_name: str):
    """Select the correct transformer class by model name."""
    lower = model_name.lower()
    if "qwen3" in lower:
        return _QWEN3_CLS or _FALLBACK_CLS
    if "qwen2.5" in lower or "qwen2_5" in lower:
        return _QWEN2_5_CLS or _FALLBACK_CLS
    if "qwen2" in lower or "qwen-vl" in lower:
        return _QWEN2_CLS or _FALLBACK_CLS
    return _FALLBACK_CLS


def _is_fp8_model(name: str) -> bool:
    lower = name.lower()
    return "fp8" in lower or "f8e4m3" in lower


def _triton_works() -> bool:
    if os.name != "nt":
        return True
    try:
        import triton  # noqa: F401
        from triton.backends.nvidia.driver import CudaUtils  # noqa: F401
        return True
    except Exception:
        return False


def _quant_config(model_name: str, quant: Quantization):
    """Return (BitsAndBytesConfig | None, dtype | None)."""
    info = HF_ALL_MODELS.get(model_name, {})
    if info.get("quantized"):
        return None, None
    if quant == Quantization.Q4:
        compute = torch.bfloat16 if is_bf16_supported() else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=compute,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        ), None
    if quant == Quantization.Q8:
        return BitsAndBytesConfig(load_in_8bit=True), None
    return None, get_optimal_dtype()


# ---------------------------------------------------------------------------
# Progress / interrupt
# ---------------------------------------------------------------------------

class _InterruptCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return _COMFY and comfy.model_management.processing_interrupted()


class _ProgressStreamer:
    def __init__(self, max_tokens: int, tokenizer=None, node_id=None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.generated = 0
        self._first = True
        self._done = False
        self.pbar = comfy.utils.ProgressBar(max_tokens, node_id=node_id) if _COMFY else None
        self.tqdm_bar = tqdm(total=max_tokens, desc="Generating", unit="token", leave=True)

    def put(self, value):
        if self._done:
            return
        if self._first:
            self._first = False
            return
        if _COMFY and comfy.model_management.processing_interrupted():
            self._done = True
            return
        n = value.shape[-1] if torch.is_tensor(value) and value.numel() > 0 else 1
        self.generated += n
        if self.pbar:
            self.pbar.update_absolute(self.generated, self.max_tokens)
        if self.tqdm_bar:
            self.tqdm_bar.update(n)

    def end(self):
        self._done = True
        if self.pbar:
            final = max(self.generated, 1)
            self.pbar.update_absolute(final, final)
        if self.tqdm_bar:
            rem = self.generated - self.tqdm_bar.n
            if rem > 0:
                self.tqdm_bar.update(rem)
            self.tqdm_bar.close()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class HFModelBackend:
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._signature = None
        self._sage = False
        print(f"[QwenVL-Utils] HF backend init ({self.device_info['device_type']})")

    def clear(self):
        self.model = self.processor = self.tokenizer = None
        self._signature = None
        self._sage = False
        clear_memory()

    # ------------------------------------------------------------------ load
    def load_model(self, model_name, quant_value, attention_mode, use_compile,
                   device_choice, keep_model_loaded, min_pixels=None, max_pixels=None):

        # FP8 guard on Windows
        if _is_fp8_model(model_name) and os.name == "nt" and not _triton_works():
            raise RuntimeError(
                f"[QwenVL-Utils] FP8 model '{model_name}' requires Triton which is "
                "unavailable on Windows. Use a non-FP8 model or run in Linux/WSL2."
            )

        quantization = enforce_memory_limits(
            model_name, Quantization.from_value(quant_value), self.device_info
        )
        attn_impl, use_sage = resolve_attention_mode(attention_mode)
        self._sage = use_sage
        print(f"[QwenVL-Utils] Attention: {attn_impl}" + (" + SageAttention" if use_sage else ""))

        device = (self.device_info["recommended_device"] if device_choice == "auto"
                  else normalize_device_choice(device_choice))

        sig = (model_name, quantization.value, attn_impl, device, use_compile, min_pixels, max_pixels)
        if keep_model_loaded and self.model is not None and self._signature == sig:
            return
        self.clear()

        model_path = ensure_hf_model(model_name)
        qcfg, dtype = _quant_config(model_name, quantization)

        load_kw = {
            "device_map": device if device != "auto" else "auto",
            "attn_implementation": attn_impl,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "ignore_mismatched_sizes": True,
        }
        if dtype is not None:
            load_kw["torch_dtype"] = dtype
        if qcfg is not None:
            load_kw["quantization_config"] = qcfg

        model_cls = _get_model_class(model_name)
        print(f"[QwenVL-Utils] Loading {model_name} ({quantization.value}, {model_cls.__name__})")

        try:
            self.model = model_cls.from_pretrained(model_path, **load_kw)
        except (RuntimeError, AttributeError) as exc:
            err = str(exc)
            if "size mismatch" in err or "has no attribute" in err:
                print("[QwenVL-Utils] Compatibility issue, retrying with eager attention...")
                load_kw["attn_implementation"] = "eager"
                load_kw["ignore_mismatched_sizes"] = True
                self.model = model_cls.from_pretrained(model_path, **load_kw)
            else:
                raise

        self.model = optimize_model_for_inference(self.model)
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        if use_compile:
            self.model = try_torch_compile(self.model, device)

        proc_kw = {"trust_remote_code": True}
        if min_pixels is not None:
            proc_kw["min_pixels"] = min_pixels
        if max_pixels is not None:
            proc_kw["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kw)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._signature = sig

    # -------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(self, prompt_text, image, video, frame_count, max_tokens,
                 temperature, top_p, num_beams, repetition_penalty, node_id=None):
        conversation = [{"role": "user", "content": []}]
        if image is not None:
            pil = tensor_to_pil(image)
            if pil:
                conversation[0]["content"].append({"type": "image", "image": pil})
        if video is not None:
            frames = [tensor_to_pil(f) for f in sample_video_frames(video, frame_count)]
            frames = [f for f in frames if f is not None]
            if frames:
                conversation[0]["content"].append({"type": "video", "video": frames})
        conversation[0]["content"].append({"type": "text", "text": prompt_text})

        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        proc_kw = {"text": [text_prompt], "return_tensors": "pt", "padding": True}
        imgs = [c["image"] for c in conversation[0]["content"] if c["type"] == "image"]
        vids = [c["video"] for c in conversation[0]["content"] if c["type"] == "video"]
        if imgs:
            proc_kw["images"] = imgs
        if vids:
            proc_kw["videos"] = [v for vl in vids for v in (vl if isinstance(vl, list) else [vl])]
            proc_kw["videos"] = [proc_kw["videos"]]

        processed = self.processor(**proc_kw)

        dev = next(self.model.parameters()).device
        model_inputs = {}
        for k, v in processed.items():
            if torch.is_tensor(v):
                v = v.contiguous()
                v = v.to(dev, non_blocking=(dev.type == "cuda"))
            model_inputs[k] = v
        if dev.type == "cuda":
            torch.cuda.current_stream().synchronize()

        stop_ids = [self.tokenizer.eos_token_id]
        if getattr(self.tokenizer, "eot_id", None) is not None:
            stop_ids.append(self.tokenizer.eot_id)

        gen_kw = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "num_return_sequences": 1,
        }
        if num_beams == 1:
            if temperature < 0.01:
                gen_kw["do_sample"] = False
            else:
                gen_kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kw.update(do_sample=False, early_stopping=True)

        streamer = None
        if _COMFY:
            streamer = _ProgressStreamer(max_tokens, self.tokenizer, node_id)
            gen_kw["streamer"] = streamer
            gen_kw["stopping_criteria"] = StoppingCriteriaList([_InterruptCriteria()])

        try:
            with SageAttentionContext(enable=self._sage):
                outputs = self.model.generate(**model_inputs, **gen_kw)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "[QwenVL-Utils] Triton compilation failed. Use a non-FP8 model on Windows."
            )
        finally:
            if streamer:
                streamer.end()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if _COMFY:
            comfy.model_management.throw_exception_if_processing_interrupted()

        input_len = model_inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------- run
    def run(self, model_name, quantization, preset_prompt, custom_prompt,
            image, video, frame_count, max_tokens, temperature, top_p,
            num_beams, repetition_penalty, seed, keep_model_loaded,
            attention_mode, use_torch_compile, device,
            min_pixels=None, max_pixels=None, unique_id=None) -> Tuple[str]:

        if _COMFY:
            comfy.model_management.throw_exception_if_processing_interrupted()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()

        self.load_model(model_name, quantization, attention_mode,
                        use_torch_compile, device, keep_model_loaded,
                        min_pixels, max_pixels)

        try:
            text = self.generate(prompt, image, video, frame_count, max_tokens,
                                 temperature, top_p, num_beams, repetition_penalty, unique_id)
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


_hf_backend_instance = None

def get_hf_backend() -> HFModelBackend:
    global _hf_backend_instance
    if _hf_backend_instance is None:
        _hf_backend_instance = HFModelBackend()
    return _hf_backend_instance
