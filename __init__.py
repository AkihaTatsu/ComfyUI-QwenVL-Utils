# ComfyUI-QwenVL-Utils
# QwenVL integration for ComfyUI (HuggingFace + GGUF)

import os
import sys

# Performance env vars (set before torch import)
os.environ.setdefault("QWENVL_MAX_COMPILE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() // 2)))
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "1")


def _check_deps():
    missing = []
    for mod, pkg in [("torch", "torch"), ("transformers", "transformers>=4.37.0"),
                     ("PIL", "Pillow"), ("numpy", "numpy")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        msg = f"[QwenVL-Utils] Missing: {', '.join(missing)}. Run: pip install -r requirements.txt"
        print(msg, file=sys.stderr)
        raise ImportError(msg)

_check_deps()

from .nodes.qwenvl_nodes import QwenVLBasic, QwenVLAdvanced
from .nodes.input_nodes import ImageLoader, VideoLoader, VideoLoaderPath
from .nodes.path_nodes import MultiplePathsInput

WEB_DIRECTORY = "./web"

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
