# ComfyUI-QwenVL-Utils
# Helper functions for nodes

import hashlib
from typing import Callable, Any, Dict, List

from PIL import ImageFile, UnidentifiedImageError

try:
    from comfy.cli_args import args
except ImportError:
    # Fallback if comfy.cli_args is not available
    class Args:
        default_hashing_function = "sha256"
    args = Args()


def conditioning_set_values(conditioning: List, values: Dict = {}) -> List:
    """Set values on conditioning data"""
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)
    return c


def pillow(fn: Callable, arg: Any) -> Any:
    """
    Safely call a PIL function with error handling for truncated images.
    
    Handles common PIL issues:
    - PIL issue #4472 (truncated images)
    - PIL issue #2445 (image identification)
    - ComfyUI issue #3416
    """
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


def hasher() -> Callable:
    """Get the configured hashing function"""
    hashfuncs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    return hashfuncs.get(
        getattr(args, 'default_hashing_function', 'sha256'), 
        hashlib.sha256
    )
