# ComfyUI-QwenVL-Utils / lib / device.py
# Device detection, memory management, dtype selection

import gc
import os
from typing import Dict, List, Any, Optional

import psutil
import torch

from .settings import HF_ALL_MODELS, Quantization

# Enable TF32 on Ampere+ GPUs
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    except Exception:
        pass


def is_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def get_optimal_dtype() -> torch.dtype:
    if is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def get_device_info() -> Dict[str, Any]:
    gpu: Dict[str, Any] = {"available": False, "total_memory": 0, "free_memory": 0, "name": "N/A"}
    device_type = "cpu"
    recommended = "cpu"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
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
        "system_memory": {"total": sys_mem.total / 1024 ** 3, "available": sys_mem.available / 1024 ** 3},
        "device_type": device_type,
        "recommended_device": recommended,
        "bf16_supported": is_bf16_supported(),
    }


def get_device_options() -> List[str]:
    return ["auto", "cpu", "mps"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def normalize_device_choice(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device == "auto":
        return "auto"
    if device.isdigit():
        device = f"cuda:{int(device)}"
    if device == "cuda" and not torch.cuda.is_available():
        print("[QwenVL-Utils] CUDA not available, falling back to CPU")
        return "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[QwenVL-Utils] CUDA not available, falling back to CPU")
            return "cpu"
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
                if idx >= torch.cuda.device_count():
                    print(f"[QwenVL-Utils] CUDA device {idx} not found, using cuda:0")
                    return "cuda:0"
            except (ValueError, IndexError):
                return "cuda:0"
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        print("[QwenVL-Utils] MPS not available, falling back to CPU")
        return "cpu"
    return device


def pick_device(device_choice: str) -> str:
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


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def enforce_memory_limits(model_name: str, quantization: Quantization, device_info: Dict) -> Quantization:
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
        needed *= 1.5
        available = device_info["system_memory"]["available"]
    else:
        available = device_info["gpu"]["free_memory"]

    if needed * 1.2 > available:
        if quantization == Quantization.FP16:
            print(f"[QwenVL-Utils] Not enough memory for FP16 ({needed:.1f}GB needed, {available:.1f}GB free). Trying 8-bit...")
            return enforce_memory_limits(model_name, Quantization.Q8, device_info)
        if quantization == Quantization.Q8:
            print(f"[QwenVL-Utils] Not enough memory for 8-bit ({needed:.1f}GB needed, {available:.1f}GB free). Trying 4-bit...")
            return Quantization.Q4
        raise RuntimeError(
            f"[QwenVL-Utils] Insufficient memory for {model_name} even at 4-bit "
            f"({needed:.1f}GB needed, {available:.1f}GB free). "
            "Try a smaller model or close other applications."
        )
    return quantization
