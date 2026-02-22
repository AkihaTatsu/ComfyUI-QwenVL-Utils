# ComfyUI-QwenVL-Utils
# Utility functions for device management, memory optimization, and helpers
# Optimized for maximum inference speed

import gc
import base64
import io
import inspect
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import numpy as np
import psutil
import torch
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download

import folder_paths

from .config import HF_ALL_MODELS, GGUF_VL_CATALOG, Quantization

# Performance: Enable TF32 on Ampere+ GPUs for faster computation
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    except Exception:
        pass


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information for model loading decisions"""
    gpu = {"available": False, "total_memory": 0, "free_memory": 0, "name": "N/A"}
    device_type = "cpu"
    recommended = "cpu"
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu = {
            "available": True,
            "total_memory": total,
            "free_memory": total - allocated,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
        }
        device_type = "nvidia_gpu"
        recommended = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type = "apple_silicon"
        recommended = "mps"
        gpu = {"available": True, "total_memory": 0, "free_memory": 0, "name": "Apple Silicon"}
    
    sys_mem = psutil.virtual_memory()
    return {
        "gpu": gpu,
        "system_memory": {
            "total": sys_mem.total / 1024**3,
            "available": sys_mem.available / 1024**3,
        },
        "device_type": device_type,
        "recommended_device": recommended,
        "bf16_supported": is_bf16_supported(),
    }


def is_bf16_supported() -> bool:
    """Check if BF16 is supported on the current hardware"""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # Ampere or newer
    except Exception:
        return False


def normalize_device_choice(device: str) -> str:
    """Normalize and validate device choice"""
    device = (device or "auto").strip().lower()
    
    if device == "auto":
        return "auto"
    
    if device.isdigit():
        device = f"cuda:{int(device)}"
    
    if device == "cuda":
        if not torch.cuda.is_available():
            print("[QwenVL-Utils] CUDA requested but not available, falling back to CPU")
            return "cpu"
        return "cuda"
    
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[QwenVL-Utils] CUDA requested but not available, falling back to CPU")
            return "cpu"
        if ":" in device:
            try:
                device_idx = int(device.split(":", 1)[1])
                if device_idx >= torch.cuda.device_count():
                    print(f"[QwenVL-Utils] CUDA device {device_idx} not available, using cuda:0")
                    return "cuda:0"
            except (ValueError, IndexError):
                print(f"[QwenVL-Utils] Invalid CUDA device format '{device}', using cuda:0")
                return "cuda:0"
        return device
    
    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            print("[QwenVL-Utils] MPS requested but not available, falling back to CPU")
            return "cpu"
        return "mps"
    
    return device


def clear_memory():
    """Clear GPU and CPU memory aggressively"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Performance: Reset peak memory stats for accurate monitoring
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def enforce_memory_limits(model_name: str, quantization: Quantization, device_info: Dict) -> Quantization:
    """
    Check if the requested quantization fits in available memory.
    Auto-downgrade if necessary.
    """
    info = HF_ALL_MODELS.get(model_name, {})
    requirements = info.get("vram_requirement", {})
    
    mapping = {
        Quantization.Q4: requirements.get("4bit", 0),
        Quantization.Q8: requirements.get("8bit", 0),
        Quantization.FP16: requirements.get("full", 0),
    }
    
    needed = mapping.get(quantization, 0)
    if not needed:
        return quantization
    
    if device_info["recommended_device"] in {"cpu", "mps"}:
        needed *= 1.5  # System memory is shared
        available = device_info["system_memory"]["available"]
    else:
        available = device_info["gpu"]["free_memory"]
    
    # Add 20% safety margin
    if needed * 1.2 > available:
        if quantization == Quantization.FP16:
            print(f"[QwenVL-Utils] Insufficient memory for FP16. Need {needed:.1f}GB, have {available:.1f}GB")
            print("[QwenVL-Utils] Auto-switching to 8-bit quantization...")
            return enforce_memory_limits(model_name, Quantization.Q8, device_info)
        if quantization == Quantization.Q8:
            print(f"[QwenVL-Utils] Insufficient memory for 8-bit. Need {needed:.1f}GB, have {available:.1f}GB")
            print("[QwenVL-Utils] Auto-switching to 4-bit quantization...")
            return Quantization.Q4
        
        error_msg = (
            f"\n" + "="*70 + "\n"
            f"[QwenVL-Utils] ERROR: Insufficient memory for {model_name}\n"
            "="*70 + "\n"
            f"Required: ~{needed:.1f}GB (with 20% safety margin: {needed * 1.2:.1f}GB)\n"
            f"Available: {available:.1f}GB\n\n"
            "Suggestions:\n"
            "  1. Close other applications to free memory\n"
            "  2. Use a smaller model (e.g., 2B/4B instead of 8B/32B)\n"
            "  3. Use GGUF quantized models (Q4_K_M, Q5_K_M)\n"
            "  4. Enable system memory offloading (slower but works)\n"
            "  5. Restart ComfyUI to clear memory leaks\n"
            "="*70
        )
        raise RuntimeError(error_msg)
    
    return quantization


def get_model_save_path() -> Path:
    """Get the model save path, following ComfyUI-QwenVL conventions"""
    # Use ComfyUI's multi-path system if available
    llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    if llm_paths:
        return Path(llm_paths[0]) / "Qwen-VL"
    
    # Fallback to default location
    return Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"


def ensure_hf_model(model_name: str) -> str:
    """Download HuggingFace model if not present, return local path"""
    info = HF_ALL_MODELS.get(model_name)
    if not info:
        known = sorted(HF_ALL_MODELS.keys())
        known_preview = "\n".join(f"  - {k}" for k in known[:10])
        if len(known) > 10:
            known_preview += f"\n  ... and {len(known) - 10} more"
        raise ValueError(
            f"\n" + "="*70 + "\n"
            f"[QwenVL-Utils] ERROR: HuggingFace model not found in configuration\n"
            "="*70 + "\n"
            f"Requested model: {model_name}\n\n"
            "This model name is not listed in hf_models.json.\n\n"
            "Possible causes:\n"
            "  1. The model name was misspelled or renamed\n"
            "  2. The hf_models.json config was changed or reset\n"
            f"  3. If this is a GGUF model, make sure the node uses '[GGUF] ' prefix\n\n"
            "Available HuggingFace models:\n"
            f"{known_preview}\n"
            "="*70
        )
    
    repo_id = info["repo_id"]
    models_dir = get_model_save_path()
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / repo_id.split("/")[-1]
    
    # Check if already downloaded (has weights)
    if target.exists() and target.is_dir():
        if any(target.glob("*.safetensors")) or any(target.glob("*.bin")):
            return str(target)
    
    print(f"[QwenVL-Utils] Downloading {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", ".git*"],
    )
    return str(target)


def get_gguf_base_dir() -> Path:
    """Get the base directory for GGUF models"""
    base_dir_value = GGUF_VL_CATALOG.get("base_dir", "LLM/GGUF")
    base_dir = Path(base_dir_value)
    
    if base_dir.is_absolute():
        return base_dir
    
    return Path(folder_paths.models_dir) / base_dir


def safe_dirname(value: str) -> str:
    """Create a safe directory name from a string"""
    value = (value or "").strip()
    if not value:
        return "unknown"
    return "".join(ch for ch in value if ch.isalnum() or ch in "._- ").strip() or "unknown"


def download_gguf_file(repo_ids: List[str], filename: str, target_path: Path):
    """Download a single GGUF file from HuggingFace"""
    if target_path.exists():
        print(f"[QwenVL-Utils] Using cached file: {target_path}")
        return
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    last_exc = None
    for repo_id in repo_ids:
        print(f"[QwenVL-Utils] Downloading {filename} from {repo_id} -> {target_path}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(target_path.parent),
                local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                downloaded_path.replace(target_path)
            if target_path.exists():
                print(f"[QwenVL-Utils] Download complete: {target_path}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"[QwenVL-Utils] Download failed from {repo_id}: {exc}")
    else:
        error_msg = (
            f"\n" + "="*70 + "\n"
            f"[QwenVL-Utils] ERROR: Failed to download {filename}\n"
            "="*70 + "\n"
            f"Last error: {last_exc}\n\n"
            "Possible causes:\n"
            "  1. No internet connection\n"
            "  2. HuggingFace Hub is down\n"
            "  3. File not found in repository\n"
            "  4. Authentication required (set HF_TOKEN env var)\n"
            "  5. Disk space full\n\n"
            "Tried repositories:\n"
        )
        for repo_id in repo_ids:
            error_msg += f"  - {repo_id}\n"
        error_msg += "\nManual download:\n"
        error_msg += f"  Visit: https://huggingface.co/{repo_ids[0]}/tree/main\n"
        error_msg += f"  Place file in: {target_path.parent}\n"
        error_msg += "="*70
        raise FileNotFoundError(error_msg) from last_exc
    
    if not target_path.exists():
        error_msg = (
            f"\n[QwenVL-Utils] ERROR: File not found after download: {target_path}\n"
            "This may indicate a file system issue or incomplete download.\n"
            "Try manually downloading the file to the target path.\n"
        )
        raise FileNotFoundError(error_msg)


def tensor_to_pil(tensor: torch.Tensor) -> Optional[Image.Image]:
    """Convert a tensor to PIL Image"""
    if tensor is None:
        return None
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def tensor_to_base64_png(tensor: torch.Tensor) -> Optional[str]:
    """Convert a tensor to base64-encoded PNG"""
    if tensor is None:
        return None
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    pil_img = Image.fromarray(array, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def sample_video_frames(video: torch.Tensor, frame_count: int) -> List[torch.Tensor]:
    """Sample frames from a video tensor"""
    if video is None:
        return []
    if video.ndim != 4:
        return [video]
    
    total = int(video.shape[0])
    frame_count = max(int(frame_count), 1)
    
    if total <= frame_count:
        return [video[i] for i in range(total)]
    
    idx = np.linspace(0, total - 1, frame_count, dtype=int)
    return [video[i] for i in idx]


def filter_kwargs_for_callable(fn: Callable, kwargs: Dict) -> Dict:
    """Filter kwargs to only include parameters accepted by a function"""
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)
    
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)
    
    allowed = set()
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(p.name)
    
    return {k: v for k, v in kwargs.items() if k in allowed}


def pick_device(device_choice: str) -> str:
    """Pick the best available device based on user choice"""
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    if device_choice.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if device_choice == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"


def get_device_options() -> List[str]:
    """Get list of available device options for UI"""
    num_gpus = torch.cuda.device_count()
    gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
    return ["auto", "cpu", "mps"] + gpu_list


def get_optimal_dtype():
    """Get the optimal dtype for the current hardware"""
    if is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    else:
        return torch.float32


def optimize_model_for_inference(model):
    """Apply comprehensive inference optimizations to a model"""
    model.eval()
    
    # Enable cache for faster generation
    if hasattr(model, 'config'):
        model.config.use_cache = True
        # Performance: Set optimal generation settings
        if hasattr(model.config, 'pretraining_tp'):
            model.config.pretraining_tp = 1
        # Enable memory efficient attention if available
        if hasattr(model.config, '_attn_implementation_autoset'):
            model.config._attn_implementation_autoset = False
    
    if hasattr(model, 'generation_config'):
        model.generation_config.use_cache = True
    
    # Performance: Disable gradient computation globally for this model
    for param in model.parameters():
        param.requires_grad = False
    
    # Performance: Fuse operations if available (PyTorch 2.0+)
    try:
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')  # Use TensorCores for faster computation
    except Exception:
        pass
    
    # Performance: Try to convert model to optimal inference dtype if on CUDA
    if torch.cuda.is_available():
        try:
            # Ensure model uses consistent dtype for optimal performance
            device = next(model.parameters()).device
            if device.type == 'cuda':
                # Most parameters should already be in optimal dtype from loading
                # This is just a safety check
                pass
        except Exception:
            pass
    
    return model


def try_torch_compile(model, device: str):
    """Attempt to apply torch.compile for significant performance gains"""
    if not device.startswith("cuda"):
        return model
    
    if not torch.cuda.is_available():
        return model
    
    try:
        # Check PyTorch version
        major, minor = torch.__version__.split('.')[:2]
        if int(major) < 2:
            print("[QwenVL-Utils] torch.compile requires PyTorch 2.0+")
            return model
        
        # Performance: Use 'max-autotune' for best performance (slower first run, ~2-3x faster inference)
        # Environment variable QWENVL_MAX_COMPILE="1" is set in __init__.py for maximum optimization
        compile_mode = "max-autotune" if os.environ.get("QWENVL_MAX_COMPILE", "0") == "1" else "reduce-overhead"
        
        # Performance: Try fullgraph=True first for maximum optimization
        # This enables aggressive graph fusion but may fail on very complex models
        print(f"[QwenVL-Utils] Applying torch.compile (mode={compile_mode})...")
        
        try:
            # Attempt fullgraph compilation for best performance
            compiled = torch.compile(
                model, 
                mode=compile_mode, 
                fullgraph=True,  # Maximum optimization
                dynamic=False     # Static shapes for better optimization
            )
            print(f"[QwenVL-Utils] ✓ torch.compile enabled with fullgraph (mode={compile_mode})")
            print(f"[QwenVL-Utils]   Note: First inference will be slower due to compilation")
            return compiled
        except Exception as e_fullgraph:
            # Fallback to partial graph compilation
            print(f"[QwenVL-Utils] Fullgraph compilation failed, using partial graph...")
            try:
                compiled = torch.compile(
                    model, 
                    mode=compile_mode, 
                    fullgraph=False,  # Partial graph optimization
                    dynamic=True      # Handle dynamic shapes
                )
                print(f"[QwenVL-Utils] ✓ torch.compile enabled (mode={compile_mode}, partial graph)")
                return compiled
            except Exception as e_partial:
                # Final fallback to simplest mode
                try:
                    compiled = torch.compile(model, mode="default")
                    print("[QwenVL-Utils] ✓ torch.compile enabled (default mode)")
                    return compiled
                except Exception as e_default:
                    print(f"[QwenVL-Utils] torch.compile failed: {e_default}")
                    return model
    except Exception as exc:
        print(f"[QwenVL-Utils] torch.compile skipped: {exc}")
        return model
