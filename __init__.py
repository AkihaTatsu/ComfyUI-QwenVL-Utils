# ComfyUI-QwenVL-Utils
# A comprehensive QwenVL integration for ComfyUI
#
# Supports:
# - HuggingFace models (Qwen3-VL, Qwen2.5-VL series)
# - GGUF models via llama-cpp-python
# - SageAttention for optimized inference
# - Flash Attention 2 and SDPA backends
#
# Models are saved to the same location as ComfyUI-QwenVL for compatibility.

import sys

def check_dependencies():
    """Check and report missing dependencies at startup"""
    missing = []
    warnings = []
    
    # Core dependencies
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA not available - CPU inference only (slower)")
    except ImportError:
        missing.append("torch - Install: pip install torch torchvision")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers - Install: pip install transformers>=4.37.0")
    
    try:
        import PIL
    except ImportError:
        missing.append("Pillow - Install: pip install Pillow")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy - Install: pip install numpy")
    
    # Optional but recommended
    try:
        import bitsandbytes
    except ImportError:
        warnings.append("bitsandbytes not installed - 4-bit/8-bit quantization unavailable")
        warnings.append("  Install: pip install bitsandbytes>=0.41.0")
    
    try:
        import flash_attn
    except ImportError:
        warnings.append("flash-attn not installed - Flash Attention 2 unavailable")
        warnings.append("  Install: pip install flash-attn --no-build-isolation")
    
    # Report status
    if missing:
        error_msg = (
            "\n" + "="*70 + "\n"
            "[QwenVL-Utils] ERROR: Missing required dependencies\n"
            "="*70 + "\n"
        )
        for dep in missing:
            error_msg += f"  ✗ {dep}\n"
        error_msg += "\nInstall all required packages:\n"
        error_msg += "  pip install -r requirements.txt\n"
        error_msg += "="*70 + "\n"
        print(error_msg, file=sys.stderr)
        raise ImportError(error_msg)
    
    if warnings:
        print("[QwenVL-Utils] Dependency warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
        print()

# Check dependencies on import
check_dependencies()

from .nodes import QwenVLBasic, QwenVLAdvanced
from .util_nodes import ImageLoader, VideoLoader, VideoLoaderPath
from .path_nodes import MultiplePathsInput

WEB_DIRECTORY = "./web"

# Node class mappings
NODE_CLASS_MAPPINGS = {
    # Main QwenVL nodes
    "QwenVL_Basic": QwenVLBasic,
    "QwenVL_Advanced": QwenVLAdvanced,
    # Input utility nodes
    "QwenVLUtils_ImageLoader": ImageLoader,
    "QwenVLUtils_VideoLoader": VideoLoader,
    "QwenVLUtils_VideoLoaderPath": VideoLoaderPath,
    "QwenVLUtils_MultiplePathsInput": MultiplePathsInput,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    # Main QwenVL nodes
    "QwenVL_Basic": "QwenVL (Basic)",
    "QwenVL_Advanced": "QwenVL (Advanced)",
    # Input utility nodes
    "QwenVLUtils_ImageLoader": "Load Image Advanced",
    "QwenVLUtils_VideoLoader": "Load Video Advanced",
    "QwenVLUtils_VideoLoaderPath": "Load Video Advanced (Path)",
    "QwenVLUtils_MultiplePathsInput": "Multiple Paths Input",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
