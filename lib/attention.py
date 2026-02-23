# ComfyUI-QwenVL-Utils / lib / attention.py
# Attention backend detection and SageAttention context manager

import torch
from typing import Tuple

ATTENTION_MODES = [
    "auto",
    "flash_attention_2",
    "sdpa_flash",
    "sage_attention",
    "sdpa_math",
    "eager",
    "sdpa",
]

# Cached availability flags
_FLASH_ATTN_AVAILABLE = None
_FLASH_SDPA_AVAILABLE = None
_SAGE_ATTN_AVAILABLE = None
_SDPA_AVAILABLE = None


def _sdpa_available() -> bool:
    global _SDPA_AVAILABLE
    if _SDPA_AVAILABLE is None:
        _SDPA_AVAILABLE = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    return _SDPA_AVAILABLE


def _flash_sdpa_available() -> bool:
    global _FLASH_SDPA_AVAILABLE
    if _FLASH_SDPA_AVAILABLE is not None:
        return _FLASH_SDPA_AVAILABLE
    if not torch.cuda.is_available():
        _FLASH_SDPA_AVAILABLE = False
        return False
    try:
        if hasattr(torch.nn.functional, "scaled_dot_product_attention") and \
           hasattr(torch.backends.cuda, "flash_sdp_enabled"):
            _FLASH_SDPA_AVAILABLE = torch.backends.cuda.flash_sdp_enabled()
            return _FLASH_SDPA_AVAILABLE
    except Exception:
        pass
    _FLASH_SDPA_AVAILABLE = False
    return False


def _flash_attn_available() -> bool:
    global _FLASH_ATTN_AVAILABLE
    if _FLASH_ATTN_AVAILABLE is not None:
        return _FLASH_ATTN_AVAILABLE
    if not torch.cuda.is_available():
        _FLASH_ATTN_AVAILABLE = False
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            _FLASH_ATTN_AVAILABLE = False
            return False
    except Exception:
        _FLASH_ATTN_AVAILABLE = False
        return False
    try:
        import flash_attn  # noqa: F401
        _FLASH_ATTN_AVAILABLE = True
    except ImportError:
        _FLASH_ATTN_AVAILABLE = False
    return _FLASH_ATTN_AVAILABLE


def _sage_attn_available() -> bool:
    global _SAGE_ATTN_AVAILABLE
    if _SAGE_ATTN_AVAILABLE is not None:
        return _SAGE_ATTN_AVAILABLE
    if not torch.cuda.is_available():
        _SAGE_ATTN_AVAILABLE = False
        return False
    try:
        from sageattention import sageattn  # noqa: F401
        _SAGE_ATTN_AVAILABLE = True
    except Exception:
        _SAGE_ATTN_AVAILABLE = False
    return _SAGE_ATTN_AVAILABLE


def resolve_attention_mode(mode: str) -> Tuple[str, bool]:
    """Resolve attention mode → (attn_implementation, use_sage_attention)."""
    if mode == "eager":
        return "eager", False

    if mode == "sdpa_flash":
        if _sdpa_available() and _flash_sdpa_available():
            return "sdpa", False
        print("[QwenVL-Utils] Flash SDPA unavailable, falling back")
        return ("sdpa", False) if _sdpa_available() else ("eager", False)

    if mode == "sdpa_math":
        return ("sdpa", False) if _sdpa_available() else ("eager", False)

    if mode == "sdpa":
        return ("sdpa", False) if _sdpa_available() else ("eager", False)

    if mode == "flash_attention_2":
        if _flash_attn_available():
            return "flash_attention_2", False
        if _sdpa_available() and _flash_sdpa_available():
            print("[QwenVL-Utils] flash-attn not installed, using built-in Flash SDPA")
            return "sdpa", False
        mode = "auto"

    if mode == "sage_attention":
        if _sage_attn_available():
            return ("sdpa", True) if _sdpa_available() else ("eager", True)
        mode = "auto"

    # Auto: flash_attention_2 > sdpa+flash > sage > sdpa > eager
    if _flash_attn_available():
        return "flash_attention_2", False
    if _sdpa_available() and _flash_sdpa_available():
        return "sdpa", False
    if _sage_attn_available():
        return ("sdpa", True) if _sdpa_available() else ("eager", True)
    if _sdpa_available():
        return "sdpa", False
    return "eager", False


class SageAttentionContext:
    """Context manager that patches torch SDPA with SageAttention."""
    _original_sdpa = None
    _patch_count = 0

    def __init__(self, enable: bool = True):
        self.enable = enable
        self._did_patch = False

    def __enter__(self):
        if not self.enable or not _sage_attn_available():
            return self
        if SageAttentionContext._patch_count > 0:
            SageAttentionContext._patch_count += 1
            self._did_patch = True
            return self
        try:
            from sageattention import sageattn
            if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                return self
            SageAttentionContext._original_sdpa = torch.nn.functional.scaled_dot_product_attention
            original = SageAttentionContext._original_sdpa

            def wrapper(query, key, value, attn_mask=None, dropout_p=0.0,
                        is_causal=False, scale=None, enable_gqa=False):
                try:
                    if attn_mask is None and dropout_p == 0.0:
                        return sageattn(query, key, value, is_causal=is_causal, smooth_k=True)
                    return original(query, key, value, attn_mask=attn_mask,
                                    dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                except Exception:
                    return original(query, key, value, attn_mask=attn_mask,
                                    dropout_p=dropout_p, is_causal=is_causal, scale=scale)

            torch.nn.functional.scaled_dot_product_attention = wrapper
            SageAttentionContext._patch_count = 1
            self._did_patch = True
        except Exception as exc:
            print(f"[QwenVL-Utils] SageAttention enable failed: {exc}")
        return self

    def __exit__(self, *_):
        if not self._did_patch:
            return False
        SageAttentionContext._patch_count -= 1
        if SageAttentionContext._patch_count == 0 and SageAttentionContext._original_sdpa is not None:
            torch.nn.functional.scaled_dot_product_attention = SageAttentionContext._original_sdpa
            SageAttentionContext._original_sdpa = None
        return False
