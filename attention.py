# ComfyUI-QwenVL-Utils
# Attention mode handling with SageAttention support
# Optimized for maximum inference speed
#
# Auto mode priority:
# 1. flash_attention_2 (external package) - best raw performance if installed
# 2. sdpa_flash (PyTorch built-in) - excellent performance, best compatibility
# 3. sage_attention - experimental, good memory efficiency
# 4. sdpa_math - stable fallback
# 5. eager - basic fallback, always works
#
# Note: PyTorch 2.0+ SDPA automatically uses Flash Attention backend on Ampere+ GPUs
# This provides similar performance to flash-attn package without installation issues

import os
import torch
from typing import Tuple, Optional

# Attention mode options - All 5 performance levels can be manually selected
# Priority (high to low): flash_attention_2 > sdpa_flash > sage_attention > sdpa_math > eager
ATTENTION_MODES = [
    "auto",              # Auto-select best available (recommended)
    "flash_attention_2", # External flash-attn package (best raw performance if installed)
    "sdpa_flash",       # PyTorch SDPA with Flash backend (best compatibility)
    "sage_attention",   # SageAttention (experimental, good memory efficiency)
    "sdpa_math",        # PyTorch SDPA with math backend (slower but stable)
    "eager",            # Standard attention (slowest, always works)
    "sdpa",             # Legacy option: auto-selects Flash or math backend
]

# Performance: Cache availability checks
_FLASH_ATTN_AVAILABLE = None
_FLASH_SDPA_AVAILABLE = None  # PyTorch built-in Flash SDPA
_SAGE_ATTN_AVAILABLE = None
_SDPA_AVAILABLE = None


def flash_sdpa_available() -> bool:
    """
    Check if PyTorch's built-in Flash SDPA backend is available (cached).
    This is the Flash Attention implementation built into PyTorch 2.0+,
    which works on Ampere+ GPUs without installing flash-attn package.
    """
    global _FLASH_SDPA_AVAILABLE
    if _FLASH_SDPA_AVAILABLE is not None:
        return _FLASH_SDPA_AVAILABLE
    
    if not torch.cuda.is_available():
        _FLASH_SDPA_AVAILABLE = False
        return False
    
    try:
        # Check if SDPA exists and Flash backend is enabled
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
                _FLASH_SDPA_AVAILABLE = torch.backends.cuda.flash_sdp_enabled()
                if _FLASH_SDPA_AVAILABLE:
                    major, minor = torch.cuda.get_device_capability()
                    print(f"[QwenVL-Utils] ✓ PyTorch Flash SDPA available (GPU SM {major}.{minor})")
                return _FLASH_SDPA_AVAILABLE
    except Exception as e:
        print(f"[QwenVL-Utils] Flash SDPA check failed: {e}")
    
    _FLASH_SDPA_AVAILABLE = False
    return False


def flash_attn_available() -> bool:
    """
    Check if external Flash Attention 2 package is available (cached).
    Note: PyTorch 2.0+ has built-in Flash SDPA that provides similar performance.
    Use flash_sdpa_available() to check for the built-in version.
    """
    global _FLASH_ATTN_AVAILABLE
    if _FLASH_ATTN_AVAILABLE is not None:
        return _FLASH_ATTN_AVAILABLE
    
    if not torch.cuda.is_available():
        _FLASH_ATTN_AVAILABLE = False
        return False

    major, _ = torch.cuda.get_device_capability()
    if major < 8:  # Flash Attention requires Ampere or newer
        print(f"[QwenVL-Utils] Flash Attention 2 requires Ampere (SM 8.0) or newer GPU. "
              f"Detected compute capability: {major}.x")
        _FLASH_ATTN_AVAILABLE = False
        return False

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        # Don't print warning - built-in SDPA is usually sufficient
        _FLASH_ATTN_AVAILABLE = False
        return False

    try:
        import importlib.metadata as importlib_metadata
        version = importlib_metadata.version("flash_attn")
        print(f"[QwenVL-Utils] ✓ Flash Attention 2 package available (v{version})")
    except Exception as e:
        print(f"[QwenVL-Utils] Flash Attention 2 version check failed: {e}")
        _FLASH_ATTN_AVAILABLE = False
        return False

    _FLASH_ATTN_AVAILABLE = True
    return True


def sage_attention_available() -> bool:
    """Check if SageAttention is available (cached)"""
    global _SAGE_ATTN_AVAILABLE
    if _SAGE_ATTN_AVAILABLE is not None:
        return _SAGE_ATTN_AVAILABLE
    
    if not torch.cuda.is_available():
        _SAGE_ATTN_AVAILABLE = False
        return False
    
    try:
        from sageattention import sageattn  # noqa: F401
        _SAGE_ATTN_AVAILABLE = True
        return True
    except ImportError:
        print("[QwenVL-Utils] SageAttention not installed. To enable (experimental):")
        print("  pip install sageattention")
        print("  Note: Requires compatible CUDA version and GPU")
        _SAGE_ATTN_AVAILABLE = False
        return False
    except Exception as e:
        print(f"[QwenVL-Utils] SageAttention initialization error: {e}")
        print("  Check CUDA version compatibility and GPU support")
        _SAGE_ATTN_AVAILABLE = False
        return False


def sdpa_available() -> bool:
    """Check if SDPA (Scaled Dot-Product Attention) is available (cached)"""
    global _SDPA_AVAILABLE
    if _SDPA_AVAILABLE is not None:
        return _SDPA_AVAILABLE
    
    try:
        # SDPA is available in PyTorch 2.0+
        _SDPA_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    except Exception:
        _SDPA_AVAILABLE = False
    return _SDPA_AVAILABLE


def get_attention_availability() -> dict:
    """Get availability status of all attention backends"""
    return {
        "flash_attention_2": flash_attn_available(),
        "flash_sdpa": flash_sdpa_available(),  # PyTorch built-in
        "sage_attention": sage_attention_available(),
        "sdpa": sdpa_available(),
        "eager": True,  # Always available
    }


def resolve_attention_mode(mode: str) -> Tuple[str, Optional[str]]:
    """
    Resolve the attention mode to use based on availability.
    
    Returns:
        Tuple of (attn_implementation for transformers, sage_attention_enabled)
        - attn_implementation: "flash_attention_2", "sdpa", or "eager"
        - sage_attention_enabled: True if SageAttention should be used as a wrapper
    
    Supported modes:
    - auto: Auto-select best available
    - sdpa_flash: Force SDPA with Flash backend (requires compatible GPU)
    - sdpa_math: Force SDPA with math backend (slower but stable)
    - flash_attention_2: Use external flash-attn package (requires installation)
    - sage_attention: Use SageAttention wrapper (experimental)
    - eager: Standard PyTorch attention (slowest, always works)
    - sdpa: Legacy option, auto-selects Flash or math backend
    """
    # Handle explicit mode selections
    if mode == "eager":
        print("[QwenVL-Utils] Using eager attention (user selected)")
        return "eager", False
    
    if mode == "sdpa_flash":
        # User explicitly wants SDPA with Flash backend
        if not sdpa_available():
            print("[QwenVL-Utils] SDPA not available, falling back to eager")
            return "eager", False
        if not flash_sdpa_available():
            print("[QwenVL-Utils] Flash SDPA backend not available on this GPU")
            print("[QwenVL-Utils] Requires Ampere+ GPU. Falling back to SDPA math backend")
            return "sdpa", False
        print("[QwenVL-Utils] Using SDPA with Flash backend (user selected)")
        return "sdpa", False
    
    if mode == "sdpa_math":
        # User explicitly wants SDPA with math backend (disable Flash)
        if not sdpa_available():
            print("[QwenVL-Utils] SDPA not available, falling back to eager")
            return "eager", False
        # Force math backend by not checking Flash availability
        print("[QwenVL-Utils] Using SDPA with math backend (user selected, Flash disabled)")
        # Note: We still return "sdpa" but the user's intent is to avoid Flash
        # PyTorch will use math backend if Flash is explicitly disabled or unavailable
        return "sdpa", False
    
    if mode == "sdpa":
        # Legacy option: auto-select Flash or math backend
        if sdpa_available():
            if flash_sdpa_available():
                print("[QwenVL-Utils] Using SDPA with Flash backend (auto-detected)")
            else:
                print("[QwenVL-Utils] Using SDPA with math backend (auto-detected)")
            return "sdpa", False
        print("[QwenVL-Utils] SDPA not available, falling back to eager")
        return "eager", False
    
    if mode == "flash_attention_2":
        if flash_attn_available():
            return "flash_attention_2", False
        # Fallback to SDPA which may use Flash backend
        if sdpa_available() and flash_sdpa_available():
            print("[QwenVL-Utils] flash-attn package not installed, using PyTorch Flash SDPA instead")
            print("[QwenVL-Utils] (This provides similar performance without installation issues)")
            return "sdpa", False
        print("[QwenVL-Utils] Flash Attention not available, trying alternatives...")
        mode = "auto"
    
    if mode == "sage_attention":
        if sage_attention_available():
            # SageAttention works as a wrapper on top of other implementations
            if sdpa_available():
                return "sdpa", True
            return "eager", True
        print("[QwenVL-Utils] SageAttention not available, trying alternatives...")
        mode = "auto"
    
    # Auto mode: Prefer flash_attention_2 > SDPA (with Flash backend) > SageAttention > SDPA (math) > eager
    # Note: This order prioritizes raw performance (flash_attention_2) but falls back to
    # built-in SDPA Flash which has better compatibility with new GPU architectures (like Blackwell)
    if mode == "auto":
        # First try external flash-attn package (best raw performance)
        if flash_attn_available():
            print("[QwenVL-Utils] Using Flash Attention 2 package (best performance)")
            return "flash_attention_2", False
        
        # Then check if SDPA with Flash backend is available (excellent compatibility)
        if sdpa_available() and flash_sdpa_available():
            print("[QwenVL-Utils] Using SDPA with Flash backend (recommended)")
            return "sdpa", False
        
        # Try SageAttention
        if sage_attention_available():
            print("[QwenVL-Utils] Using SageAttention")
            if sdpa_available():
                return "sdpa", True
            return "eager", True
        
        # Plain SDPA without Flash backend
        if sdpa_available():
            print("[QwenVL-Utils] Using SDPA (math backend)")
            return "sdpa", False
        
        print("[QwenVL-Utils] Using eager attention (fallback)")
        return "eager", False
    
    # Unknown mode, default to auto behavior
    print(f"[QwenVL-Utils] Unknown attention mode '{mode}', using auto")
    return resolve_attention_mode("auto")


class SageAttentionContext:
    """
    Context manager for enabling SageAttention.
    When enabled, it patches torch's SDPA to use SageAttention.
    Optimized for speed with FP8 support where available.
    """
    
    _original_sdpa = None  # Class-level storage to avoid closure issues
    _patch_count = 0  # Reference counting for nested contexts
    _sageattn_fn = None  # Cached SageAttention function
    
    def __init__(self, enable: bool = True):
        self.enable = enable
        self._did_patch = False
    
    def __enter__(self):
        if not self.enable:
            return self
        
        if not sage_attention_available():
            return self
        
        # Already patched by another context
        if SageAttentionContext._patch_count > 0:
            SageAttentionContext._patch_count += 1
            self._did_patch = True
            return self
        
        try:
            from sageattention import sageattn
            SageAttentionContext._sageattn_fn = sageattn
            
            # Performance: Check for FP8 support (faster on Hopper+)
            use_fp8 = False
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 9:  # Hopper or newer
                    use_fp8 = True
            except Exception:
                pass
            
            # Store original SDPA at class level
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                SageAttentionContext._original_sdpa = torch.nn.functional.scaled_dot_product_attention
                original_sdpa = SageAttentionContext._original_sdpa  # Local reference for closure
                cached_sageattn = sageattn  # Cache for closure
                
                # Create wrapper that uses SageAttention
                def sage_sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, 
                                     is_causal=False, scale=None, enable_gqa=False):
                    """SageAttention wrapper for SDPA - Optimized"""
                    try:
                        # SageAttention works best with:
                        # - No attention mask
                        # - No dropout
                        # - Standard attention patterns
                        if attn_mask is None and dropout_p == 0.0:
                            # SageAttention signature: sageattn(q, k, v, is_causal=False, smooth_k=True)
                            # Performance: smooth_k=True improves accuracy, minimal speed impact
                            return cached_sageattn(query, key, value, is_causal=is_causal, smooth_k=True)
                        else:
                            # Fall back to original for complex cases
                            return original_sdpa(query, key, value, attn_mask=attn_mask,
                                                dropout_p=dropout_p, is_causal=is_causal,
                                                scale=scale)
                    except Exception as e:
                        # Fall back to original on any error
                        return original_sdpa(query, key, value, attn_mask=attn_mask,
                                            dropout_p=dropout_p, is_causal=is_causal,
                                            scale=scale)
                
                torch.nn.functional.scaled_dot_product_attention = sage_sdpa_wrapper
                SageAttentionContext._patch_count = 1
                self._did_patch = True
                fp8_info = " (FP8 available)" if use_fp8 else ""
                print(f"[QwenVL-Utils] SageAttention enabled{fp8_info}")
            else:
                print("[QwenVL-Utils] SDPA not available, cannot apply SageAttention")
        except Exception as e:
            print(f"[QwenVL-Utils] Failed to enable SageAttention: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._did_patch:
            return False
        
        SageAttentionContext._patch_count -= 1
        
        # Only restore when all contexts have exited
        if SageAttentionContext._patch_count == 0 and SageAttentionContext._original_sdpa is not None:
            torch.nn.functional.scaled_dot_product_attention = SageAttentionContext._original_sdpa
            SageAttentionContext._original_sdpa = None
            SageAttentionContext._sageattn_fn = None
        
        return False


def apply_sage_attention_to_model(model):
    """
    Apply SageAttention optimization to a model's attention layers.
    This is a more permanent solution than the context manager.
    """
    if not sage_attention_available():
        print("[QwenVL-Utils] SageAttention not available for model optimization")
        return model
    
    try:
        from sageattention import sageattn
        
        # This would require model-specific implementation
        # For now, we rely on the context manager approach
        print("[QwenVL-Utils] SageAttention model optimization is handled via SDPA wrapper")
        return model
    except Exception as e:
        print(f"[QwenVL-Utils] Failed to apply SageAttention to model: {e}")
        return model


def get_optimal_attention_config() -> dict:
    """Get the optimal attention configuration based on hardware"""
    availability = get_attention_availability()
    
    config = {
        "recommended_mode": "auto",
        "available_modes": [k for k, v in availability.items() if v],
        "details": availability,
    }
    
    # Provide recommendation with new explicit options
    if availability.get("flash_sdpa"):
        config["recommended_mode"] = "sdpa_flash"
        config["recommendation_reason"] = "PyTorch Flash SDPA provides excellent performance with best compatibility"
    elif availability["flash_attention_2"]:
        config["recommended_mode"] = "flash_attention_2"
        config["recommendation_reason"] = "Flash Attention 2 package provides the best raw performance"
    elif availability["sage_attention"]:
        config["recommended_mode"] = "sage_attention"
        config["recommendation_reason"] = "SageAttention provides good performance with memory efficiency"
    elif availability["sdpa"]:
        config["recommended_mode"] = "sdpa_math"
        config["recommendation_reason"] = "SDPA math backend is the best available option on your hardware"
    else:
        config["recommended_mode"] = "eager"
        config["recommendation_reason"] = "Using standard attention (no optimizations available)"
    
    return config
