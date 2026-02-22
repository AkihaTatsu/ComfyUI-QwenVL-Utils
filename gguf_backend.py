# ComfyUI-QwenVL-Utils
# GGUF model backend for QwenVL (via llama-cpp-python)
# Optimized for maximum inference speed

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Any

import torch

# ComfyUI progress bar and interruption support
try:
    import comfy.utils
    import comfy.model_management
    COMFY_PROGRESS_AVAILABLE = True
except ImportError:
    COMFY_PROGRESS_AVAILABLE = False

from .config import GGUF_VL_CATALOG, SYSTEM_PROMPTS
from .utils import (
    get_gguf_base_dir,
    safe_dirname,
    download_gguf_file,
    tensor_to_base64_png,
    sample_video_frames,
    pick_device,
    clear_memory,
    filter_kwargs_for_callable,
)
from .output_cleaner import clean_model_output, OutputCleanConfig

# Performance: Set llama.cpp environment variables before import
os.environ.setdefault("LLAMA_CUBLAS", "1")  # Enable CUDA acceleration
os.environ.setdefault("LLAMA_METAL", "1")   # Enable Metal on macOS


def _read_gguf_architecture(path: str) -> Optional[str]:
    """Read the 'general.architecture' value from a GGUF file header.

    Returns the architecture string (e.g. 'qwen3vl', 'llama', 'qwen2vl')
    or None if the file cannot be parsed.
    """
    import struct
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            _version = struct.unpack("<I", f.read(4))[0]
            _n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def _read_str(fh):
                length = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(length).decode("utf-8", errors="replace")

            # Scan KV pairs for general.architecture (usually first)
            for _ in range(min(n_kv, 10)):
                key = _read_str(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if vtype == 8:  # string
                    val = _read_str(f)
                elif vtype == 4:  # uint32
                    val = struct.unpack("<I", f.read(4))[0]
                elif vtype == 5:  # int32
                    val = struct.unpack("<i", f.read(4))[0]
                elif vtype == 6:  # float32
                    val = struct.unpack("<f", f.read(4))[0]
                elif vtype == 7:  # bool
                    val = struct.unpack("<?", f.read(1))[0]
                elif vtype == 10:  # uint64
                    val = struct.unpack("<Q", f.read(8))[0]
                else:
                    break  # unsupported type – stop scanning
                if key == "general.architecture":
                    return str(val) if isinstance(val, str) else None
    except Exception:
        pass
    return None


def _get_supported_architectures() -> Optional[set]:
    """Query llama.cpp for its list of supported model architectures.

    Returns a set of lowercase architecture names, or None if detection fails.
    """
    try:
        import llama_cpp.llama_cpp as _lc
        # llama_cpp exposes llama_supports_model_arch(name) since ~b3600
        if hasattr(_lc, "llama_supports_model_arch"):
            # No easy enumeration; we'll test on demand in the caller.
            return None
        # Fallback: try loading with vocab_only to see if it errors
        return None
    except Exception:
        return None


def _check_architecture_supported(model_path: str) -> Tuple[bool, str]:
    """Check whether the GGUF architecture is supported by llama.cpp.

    Returns (supported: bool, architecture: str).
    When *supported* is False, the caller should NOT attempt any fallback loads.
    """
    arch = _read_gguf_architecture(model_path)
    if arch is None:
        # Cannot determine – let the normal load path try
        return True, "unknown"

    try:
        import llama_cpp.llama_cpp as _lc
        params = _lc.llama_model_default_params()
        params.vocab_only = True  # fast check – only parse metadata
        model = _lc.llama_model_load_from_file(model_path.encode("utf-8"), params)
        if model is not None:
            _lc.llama_model_free(model)
            return True, arch
        else:
            return False, arch
    except Exception:
        return True, arch  # unsure – let normal path try


def _patch_llama_model_del():
    """Monkey-patch LlamaModel.__del__ to suppress the 'sampler' AttributeError.

    Older/mismatched builds of llama-cpp-python raise
    AttributeError: 'LlamaModel' object has no attribute 'sampler'
    during garbage collection, which floods the console with tracebacks.
    """
    try:
        from llama_cpp._internals import LlamaModel
        _original_del = getattr(LlamaModel, "__del__", None)
        if _original_del is None:
            return

        def _safe_del(self):
            try:
                _original_del(self)
            except (AttributeError, Exception):
                # Swallow cleanup errors – the object is being GC'd anyway
                try:
                    if hasattr(self, "_exit_stack"):
                        self._exit_stack.close()
                except Exception:
                    pass

        LlamaModel.__del__ = _safe_del
    except Exception:
        pass


@dataclass(frozen=True)
class GGUFModelResolved:
    """Resolved GGUF model configuration"""
    display_name: str
    repo_id: Optional[str]
    alt_repo_ids: List[str]
    author: Optional[str]
    repo_dirname: str
    model_filename: str
    mmproj_filename: Optional[str]
    context_length: int
    image_max_tokens: int
    n_batch: int
    gpu_layers: int
    top_k: int
    pool_size: int


def _flatten_gguf_catalog() -> dict:
    """Flatten the GGUF catalog into a single dict of models"""
    flattened = {}
    seen_display_names = set()
    
    repos = (
        GGUF_VL_CATALOG.get("qwenVL_model") or 
        GGUF_VL_CATALOG.get("vl_repos") or 
        GGUF_VL_CATALOG.get("repos") or 
        {}
    )
    
    for repo_key, repo in repos.items():
        if not isinstance(repo, dict):
            continue
        
        author = repo.get("author") or repo.get("publisher")
        repo_name = repo.get("repo_name") or repo.get("repo_name_override") or repo_key
        repo_id = repo.get("repo_id") or (f"{author}/{repo_name}" if author and repo_name else None)
        alt_repo_ids = repo.get("alt_repo_ids") or []
        
        defaults = repo.get("defaults") or {}
        mmproj_file = repo.get("mmproj_file")
        model_files = repo.get("model_files") or []
        
        for model_file in model_files:
            display = Path(model_file).name
            if display in seen_display_names:
                display = f"{display} ({repo_key})"
            seen_display_names.add(display)
            
            flattened[display] = {
                **defaults,
                "author": author,
                "repo_dirname": repo_name,
                "repo_id": repo_id,
                "alt_repo_ids": alt_repo_ids,
                "filename": model_file,
                "mmproj_filename": mmproj_file,
            }
    
    # Also include legacy "models" format
    legacy_models = GGUF_VL_CATALOG.get("models") or {}
    for name, entry in legacy_models.items():
        if isinstance(entry, dict):
            flattened[name] = entry
    
    return flattened


def get_gguf_vl_models() -> List[str]:
    """Get list of available GGUF VL models"""
    all_models = _flatten_gguf_catalog()
    return sorted([
        key for key, entry in all_models.items() 
        if (entry or {}).get("mmproj_filename")
    ]) or ["(no GGUF VL models configured)"]


def _resolve_model_entry(model_name: str) -> GGUFModelResolved:
    """Resolve a model name to its full configuration"""
    all_models = _flatten_gguf_catalog()
    entry = all_models.get(model_name) or {}
    
    if not entry:
        # Try to find by filename
        wanted = {model_name, f"{model_name}.gguf"}
        for candidate in all_models.values():
            filename = candidate.get("filename")
            if filename and Path(filename).name in wanted:
                entry = candidate
                break
    
    if not entry:
        # Model is completely unknown — raise a clear, actionable error immediately
        known = sorted(all_models.keys())
        known_preview = "\n".join(f"  - {k}" for k in known[:10])
        if len(known) > 10:
            known_preview += f"\n  ... and {len(known) - 10} more"
        error_msg = (
            "\n" + "="*70 + "\n"
            f"[QwenVL-Utils] ERROR: GGUF model not found in configuration\n"
            "="*70 + "\n"
            f"Requested model: {model_name}\n\n"
            "This model is not listed in gguf_models.json.\n\n"
            "Possible causes:\n"
            "  1. The model name is wrong or was renamed\n"
            "  2. The gguf_models.json config was changed or reset\n"
            "  3. The model belongs to a different plugin installation\n\n"
            "Available GGUF models:\n"
            f"{known_preview}\n\n"
            "To add this model, edit gguf_models.json and add an entry\n"
            "under the 'qwenVL_model' section.\n"
            "="*70
        )
        raise ValueError(error_msg)
    
    repo_id = entry.get("repo_id")
    alt_repo_ids = entry.get("alt_repo_ids") or []
    
    author = entry.get("author") or entry.get("publisher")
    repo_dirname = entry.get("repo_dirname") or (
        repo_id.split("/")[-1] if isinstance(repo_id, str) and "/" in repo_id else model_name
    )
    
    model_filename = entry.get("filename")
    mmproj_filename = entry.get("mmproj_filename")
    
    if not model_filename:
        raise ValueError(
            f"[QwenVL-Utils] GGUF configuration for '{model_name}' is missing the "
            "'filename' field. Please check gguf_models.json."
        )
    
    def _int(name: str, default: int) -> int:
        value = entry.get(name, default)
        try:
            return int(value)
        except Exception:
            return default
    
    return GGUFModelResolved(
        display_name=model_name,
        repo_id=repo_id,
        alt_repo_ids=[str(x) for x in alt_repo_ids if x],
        author=str(author) if author else None,
        repo_dirname=safe_dirname(str(repo_dirname)),
        model_filename=str(model_filename),
        mmproj_filename=str(mmproj_filename) if mmproj_filename else None,
        context_length=_int("context_length", 8192),
        image_max_tokens=_int("image_max_tokens", 4096),
        n_batch=_int("n_batch", 512),
        gpu_layers=_int("gpu_layers", -1),
        top_k=_int("top_k", 0),
        pool_size=_int("pool_size", 4194304),
    )


class GGUFModelBackend:
    """GGUF model backend for QwenVL inference via llama-cpp-python - Optimized for speed"""
    
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        # Cache the last failed (model_name, device) pair to avoid printing the
        # same error message repeatedly when ComfyUI re-executes the node.
        self._last_failed_model: Optional[str] = None
    
    def clear(self):
        """Clear loaded model and free memory"""
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        self._last_failed_model = None
        clear_memory()
    
    def _check_backend(self):
        """Check if llama-cpp-python is available"""
        try:
            from llama_cpp import Llama  # noqa: F401
        except ImportError as exc:
            error_msg = (
                "\n" + "="*70 + "\n"
                "[QwenVL-Utils] ERROR: llama-cpp-python not found\n"
                "="*70 + "\n"
                "GGUF models require llama-cpp-python with vision support.\n\n"
                "Installation options:\n"
                "  1. With CUDA (NVIDIA GPU) - Fastest:\n"
                "     pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n\n"
                "  2. CPU only:\n"
                "     pip install llama-cpp-python\n\n"
                "  3. Metal (Apple Silicon):\n"
                "     CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python\n\n"
                "For detailed instructions, see:\n"
                "  https://github.com/1038lab/ComfyUI-QwenVL/blob/main/docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md\n"
                "="*70
            )
            raise RuntimeError(error_msg) from exc
    
    def load_model(
        self,
        model_name: str,
        device: str,
        ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        gpu_layers: Optional[int] = None,
        image_max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        pool_size: Optional[int] = None,
    ):
        """Load GGUF model"""
        self._check_backend()
        
        resolved = _resolve_model_entry(model_name)
        base_dir = get_gguf_base_dir()
        
        author_dir = safe_dirname(resolved.author or "")
        repo_dir = safe_dirname(resolved.repo_dirname)
        target_dir = base_dir / author_dir / repo_dir
        
        model_path = target_dir / Path(resolved.model_filename).name
        mmproj_path = target_dir / Path(resolved.mmproj_filename).name if resolved.mmproj_filename else None
        
        # Prepare repo IDs for download
        repo_ids = []
        if resolved.repo_id:
            repo_ids.append(resolved.repo_id)
        repo_ids.extend(resolved.alt_repo_ids)
        
        # Download model if not present
        if not model_path.exists():
            if not repo_ids:
                error_msg = (
                    f"\n" + "="*70 + "\n"
                    f"[QwenVL-Utils] ERROR: GGUF model file not found\n"
                    "="*70 + "\n"
                    f"Model: {model_name}\n"
                    f"Expected path: {model_path}\n"
                    f"Filename: {resolved.model_filename}\n\n"
                    "No repository configured for automatic download.\n\n"
                    "Manual download required:\n"
                    "  1. Find the GGUF model file online\n"
                    "  2. Download the .gguf file\n"
                    f"  3. Place it in: {target_dir}\n"
                    f"  4. Ensure filename matches: {resolved.model_filename}\n"
                    "="*70
                )
                raise FileNotFoundError(error_msg)
            download_gguf_file(repo_ids, resolved.model_filename, model_path)
        
        # Download mmproj if needed
        if mmproj_path is not None and not mmproj_path.exists():
            if not repo_ids:
                error_msg = (
                    f"\n" + "="*70 + "\n"
                    f"[QwenVL-Utils] ERROR: GGUF mmproj file not found\n"
                    "="*70 + "\n"
                    f"Model: {model_name}\n"
                    f"Expected path: {mmproj_path}\n"
                    f"Filename: {resolved.mmproj_filename}\n\n"
                    "Vision models require both .gguf and mmproj files.\n"
                    "No repository configured for automatic download.\n\n"
                    "Manual download required:\n"
                    f"  1. Download mmproj file from model repository\n"
                    f"  2. Place it in: {target_dir}\n"
                    f"  3. Ensure filename matches: {resolved.mmproj_filename}\n"
                    "="*70
                )
                raise FileNotFoundError(error_msg)
            download_gguf_file(repo_ids, resolved.mmproj_filename, mmproj_path)
        
        device_kind = pick_device(device)
        
        # Resolve parameters
        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size
        
        n_gpu_layers = 0
        if device_kind == "cuda":
            n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
        
        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens
        
        has_mmproj = mmproj_path is not None and mmproj_path.exists()
        
        # Check signature for caching
        signature = (
            str(model_path),
            str(mmproj_path) if has_mmproj else "",
            n_ctx,
            n_batch_val,
            n_gpu_layers,
            img_max,
            top_k_val,
            pool_size_val,
        )
        
        if self.llm is not None and self.current_signature == signature:
            return
        
        self.clear()
        
        from llama_cpp import Llama
        
        # Patch LlamaModel.__del__ to suppress 'sampler' AttributeError spam
        _patch_llama_model_del()
        
        # ── Pre-validate architecture support ──
        # This avoids slow, noisy fallback attempts when the underlying
        # llama.cpp simply doesn't recognise the model architecture.
        arch_supported, arch_name = _check_architecture_supported(str(model_path))
        if not arch_supported:
            try:
                import llama_cpp as _llama_mod
                _llama_ver = getattr(_llama_mod, "__version__", "unknown")
            except Exception:
                _llama_ver = "unknown"
            raise ValueError(
                "\n" + "="*70 + "\n"
                f"[QwenVL-Utils] ERROR: Unsupported model architecture '{arch_name}'\n"
                "="*70 + "\n"
                f"Model: {model_path.name}\n"
                f"Architecture: {arch_name}\n"
                f"Installed llama-cpp-python: {_llama_ver}\n\n"
                f"Your llama-cpp-python build does not support the '{arch_name}' architecture.\n"
                "This usually means you need a newer version.\n\n"
                "Solutions (try in order):\n"
                "  1. Update llama-cpp-python to the latest release:\n"
                "     pip install --upgrade llama-cpp-python --force-reinstall --no-cache-dir\n\n"
                "  2. Install a pre-built CUDA wheel (if NVIDIA GPU):\n"
                "     pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n\n"
                "  3. Build from source with latest llama.cpp:\n"
                "     CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python "
                "--force-reinstall --no-cache-dir\n\n"
                f"The '{arch_name}' architecture requires a llama.cpp build that includes\n"
                "support for this model family.  Check the llama.cpp release notes at\n"
                "https://github.com/ggerganov/llama.cpp/releases for version compatibility.\n"
                "="*70
            )
        
        print(f"[QwenVL-Utils] GGUF architecture: {arch_name}")
        
        # ── Load chat handler for vision ──
        self.chat_handler = None
        if has_mmproj:
            handler_cls = None
            try:
                from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                handler_cls = Qwen3VLChatHandler
            except ImportError:
                try:
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                    handler_cls = Qwen25VLChatHandler
                except ImportError:
                    error_msg = (
                        "\n" + "="*70 + "\n"
                        "[QwenVL-Utils] ERROR: Qwen VL chat handler not found\n"
                        "="*70 + "\n"
                        "Your llama-cpp-python installation does not support Qwen VL models.\n\n"
                        "Solutions:\n"
                        "  1. Update llama-cpp-python to latest version:\n"
                        "     pip install --upgrade llama-cpp-python\n\n"
                        "  2. Install multimodal-capable build (recommended):\n"
                        "     pip uninstall llama-cpp-python\n"
                        "     pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n\n"
                        "  3. Check for compatible version:\n"
                        "     pip install 'llama-cpp-python>=0.2.70'\n\n"
                        "Note: Qwen3VLChatHandler or Qwen25VLChatHandler required for vision support.\n"
                        "="*70
                    )
                    raise RuntimeError(error_msg)
            
            mmproj_kwargs = {
                "clip_model_path": str(mmproj_path),
                "image_max_tokens": img_max,
                "force_reasoning": False,
                "verbose": False,
            }
            mmproj_kwargs = filter_kwargs_for_callable(
                getattr(handler_cls, "__init__", handler_cls), 
                mmproj_kwargs
            )
            if "image_max_tokens" not in mmproj_kwargs:
                print("[QwenVL-Utils] Warning: chat handler does not support image_max_tokens")
            
            self.chat_handler = handler_cls(**mmproj_kwargs)
        
        # ── Prepare Llama kwargs ──
        # Core kwargs that are safe to pass to any llama-cpp-python version
        llm_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "verbose": False,
            "use_mmap": True,
            "use_mlock": False,
            "n_threads": max(1, os.cpu_count() // 2),
            "n_threads_batch": max(1, os.cpu_count() // 2),
        }
        
        if has_mmproj and self.chat_handler is not None:
            llm_kwargs["chat_handler"] = self.chat_handler
            llm_kwargs["image_min_tokens"] = 1024
            llm_kwargs["image_max_tokens"] = img_max
        
        if device_kind == "cuda" and n_gpu_layers == 0:
            print("[QwenVL-Utils] Warning: device=cuda but n_gpu_layers=0; model runs on CPU")
        
        # ── Detect which params the Llama() constructor *explicitly* accepts ──
        # Llama.__init__ has **kwargs, so filter_kwargs_for_callable will let
        # everything through.  We must manually inspect the explicit parameter
        # list to avoid leaking unknown keys into the C backend.
        import inspect as _inspect
        try:
            _llama_sig = _inspect.signature(Llama.__init__)
            _llama_explicit = {
                p.name for p in _llama_sig.parameters.values()
                if p.kind in (
                    _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    _inspect.Parameter.KEYWORD_ONLY,
                )
            }
        except Exception:
            _llama_explicit = set()
        
        # Only add optional performance/tuning params if the constructor
        # declares them explicitly (not just via **kwargs)
        _extras = {
            "swa_full": True,
            "flash_attn": True,
            "pool_size": pool_size_val,
            "top_k": top_k_val,
        }
        _extras_added = []
        for _ek, _ev in _extras.items():
            if _ek in _llama_explicit:
                llm_kwargs[_ek] = _ev
                _extras_added.append(_ek)
        
        # Also strip any keys the constructor doesn't explicitly accept
        # (they'd fall into **kwargs and may break the C layer)
        if _llama_explicit:
            _to_remove = [
                k for k in llm_kwargs
                if k not in _llama_explicit and k != "model_path"
            ]
            for k in _to_remove:
                # Keep essential keys that Llama definitely uses
                if k in ("model_path", "n_ctx", "n_gpu_layers", "n_batch",
                         "verbose", "use_mmap", "use_mlock",
                         "n_threads", "n_threads_batch",
                         "chat_handler", "chat_format",
                         "flash_attn", "swa_full"):
                    continue
                llm_kwargs.pop(k, None)
        
        _flash = llm_kwargs.get("flash_attn", False)
        _vision = has_mmproj and self.chat_handler is not None
        print(f"[QwenVL-Utils] Loading GGUF: {model_path.name} "
              f"(device={device_kind}, gpu_layers={n_gpu_layers}, "
              f"ctx={n_ctx}, flash_attn={_flash}, vision={_vision})")
        
        # ── Progressive fallback loading ──
        # Build a chain of configurations from most to least optimised.
        _fallback_removals: List[Tuple[str, dict]] = [
            ("full", {}),
        ]
        # 1) disable flash_attn (common failure on older GPUs / builds)
        if "flash_attn" in llm_kwargs and llm_kwargs["flash_attn"]:
            _fallback_removals.append(
                ("without flash_attn", {"flash_attn": False})
            )
        # 2) disable all perf extras
        _perf_keys = [k for k in _extras_added if k in llm_kwargs]
        if _perf_keys:
            _fallback_removals.append(
                ("without " + "+".join(_perf_keys),
                 {k: None for k in _perf_keys})
            )
        # 3) halve context (OOM guard)
        if n_ctx > 2048:
            _fallback_removals.append(
                (f"ctx={n_ctx // 2}",
                 {**{k: None for k in _perf_keys}, "n_ctx": n_ctx // 2})
            )
        # 4) CPU-only
        if n_gpu_layers != 0:
            _fallback_removals.append(
                ("CPU only (gpu_layers=0)",
                 {**{k: None for k in _perf_keys}, "n_gpu_layers": 0,
                  "n_ctx": min(n_ctx, 4096)})
            )
        
        last_error = None
        for _idx, (_desc, _mods) in enumerate(_fallback_removals):
            _kw = dict(llm_kwargs)
            for _mk, _mv in _mods.items():
                if _mv is None:
                    _kw.pop(_mk, None)
                else:
                    _kw[_mk] = _mv
            
            if _idx > 0:
                _f = _kw.get("flash_attn", False)
                print(f"[QwenVL-Utils] Retry {_idx}/{len(_fallback_removals)-1}: {_desc} "
                      f"(gpu_layers={_kw.get('n_gpu_layers', 0)}, "
                      f"ctx={_kw.get('n_ctx', n_ctx)}, flash_attn={_f})")
            
            try:
                self.llm = Llama(**_kw)
                if _idx > 0:
                    print(f"[QwenVL-Utils] Model loaded successfully with fallback: {_desc}")
                last_error = None
                break
            except (ValueError, RuntimeError, OSError) as exc:
                last_error = exc
                err_str = str(exc)
                print(f"[QwenVL-Utils] Load attempt failed ({_desc}): {err_str}")
                
                # If the error is about unsupported architecture, stop immediately
                if "unknown model architecture" in err_str.lower():
                    break
                
                self.llm = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        if last_error is not None:
            try:
                import llama_cpp as _llama_mod
                _llama_ver = getattr(_llama_mod, "__version__", "unknown")
            except Exception:
                _llama_ver = "unknown"
            raise ValueError(
                f"\n" + "="*70 + "\n"
                f"[QwenVL-Utils] Failed to load GGUF model after "
                f"{len(_fallback_removals)} attempts.\n"
                "="*70 + "\n"
                f"Model: {model_path.name}\n"
                f"Architecture: {arch_name}\n"
                f"Installed llama-cpp-python: {_llama_ver}\n"
                f"Last error: {last_error}\n\n"
                "Possible causes:\n"
                f"  - Architecture '{arch_name}' not supported by this build\n"
                "  - Corrupted or incomplete GGUF file\n"
                "  - Insufficient VRAM / RAM\n"
                "  - llama-cpp-python version too old – try:\n"
                "    pip install --upgrade llama-cpp-python --force-reinstall --no-cache-dir\n"
                "="*70
            ) from last_error
        
        self.current_signature = signature
    
    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
    ) -> str:
        """Invoke the model for generation"""
        if images_b64:
            content = [{"type": "text", "text": user_prompt}]
            for img in images_b64:
                if not img:
                    continue
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{img}"}
                })
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        
        start = time.perf_counter()
        
        # Use streaming mode with progress bar if available
        if COMFY_PROGRESS_AVAILABLE:
            pbar = comfy.utils.ProgressBar(max_tokens)
            token_count = 0
            collected_content = []
            interrupted = False
            
            # Stream tokens with progress bar updates
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                repeat_penalty=float(repetition_penalty),
                seed=int(seed),
                stop=["<|im_end|>", "<|im_start|>"],
                stream=True,
            )
            
            for chunk in stream:
                # Check for interruption each token
                if comfy.model_management.processing_interrupted():
                    interrupted = True
                    break
                
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                chunk_content = delta.get("content", "")
                if chunk_content:
                    collected_content.append(chunk_content)
                    token_count += 1  # Approximate: each chunk is roughly 1 token
                    pbar.update_absolute(token_count, max_tokens)
            
            # Mark progress as complete with actual token count
            if not interrupted:
                pbar.update_absolute(token_count, token_count)
            
            elapsed = max(time.perf_counter() - start, 1e-6)
            
            # Log performance
            if token_count > 0:
                tok_s = token_count / elapsed
                status = " (interrupted)" if interrupted else ""
                print(f"[QwenVL-Utils] Tokens: completion~{token_count}{status}, "
                      f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")
            
            content = "".join(collected_content)
            
            # Re-raise interruption after clean exit from the stream
            if interrupted:
                comfy.model_management.throw_exception_if_processing_interrupted()
        else:
            # Non-streaming mode (no progress bar)
            result = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                repeat_penalty=float(repetition_penalty),
                seed=int(seed),
                stop=["<|im_end|>", "<|im_start|>"],
            )
            elapsed = max(time.perf_counter() - start, 1e-6)
            
            # Log performance
            usage = result.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(completion_tokens, int) and completion_tokens > 0:
                tok_s = completion_tokens / elapsed
                if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                    print(f"[QwenVL-Utils] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, "
                          f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")
                else:
                    print(f"[QwenVL-Utils] Tokens: completion={completion_tokens}, "
                          f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")
            
            content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        
        cleaned = clean_model_output(str(content or ""), OutputCleanConfig(mode="text"))
        return cleaned.strip()
    
    def run(
        self,
        model_name: str,
        preset_prompt: str,
        custom_prompt: str,
        image: Optional[Any],
        video: Optional[Any],
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        device: str,
        ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        gpu_layers: Optional[int] = None,
        image_max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        pool_size: Optional[int] = None,
    ) -> Tuple[str]:
        """Run inference with the model"""
        # Check for interruption before starting
        if COMFY_PROGRESS_AVAILABLE:
            comfy.model_management.throw_exception_if_processing_interrupted()
        
        # Guard: if the same model already failed to load in a previous call,
        # suppress the redundant duplicate error and raise a quiet reminder.
        # This prevents dozens of identical stack traces when ComfyUI retries.
        if self._last_failed_model == model_name:
            raise RuntimeError(
                f"[QwenVL-Utils] GGUF model '{model_name}' previously failed to load. "
                "Please check the error above and fix the issue before retrying."
            )
        
        torch.manual_seed(int(seed))
        
        # Resolve prompt
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        
        # Prepare images
        images_b64 = []
        if image is not None:
            img = tensor_to_base64_png(image)
            if img:
                images_b64.append(img)
        
        if video is not None:
            for frame in sample_video_frames(video, int(frame_count)):
                img = tensor_to_base64_png(frame)
                if img:
                    images_b64.append(img)
        
        try:
            self.load_model(
                model_name=model_name,
                device=device,
                ctx=ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                image_max_tokens=image_max_tokens,
                top_k=top_k,
                pool_size=pool_size,
            )
            # Load succeeded — clear any previous failure record for this model
            self._last_failed_model = None
            
            if images_b64 and self.chat_handler is None:
                print("[QwenVL-Utils] Warning: images provided but model has no mmproj; images ignored")
            
            text = self._invoke(
                system_prompt=(
                    "You are a helpful vision-language assistant. "
                    "Answer directly with the final answer only. No <think> and no reasoning."
                ),
                user_prompt=prompt,
                images_b64=images_b64 if self.chat_handler is not None else [],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
            return (text,)
        except Exception as exc:
            # Record that this model failed so subsequent calls can skip the
            # expensive (and noisy) download/load attempt.
            self._last_failed_model = model_name
            raise
        finally:
            if not keep_model_loaded:
                # Only clear the live model; keep _last_failed_model so the
                # guard above still works on the next call.
                self.llm = None
                self.chat_handler = None
                self.current_signature = None
                clear_memory()


# Global instance for reuse
_gguf_backend_instance = None


def get_gguf_backend() -> GGUFModelBackend:
    """Get or create the GGUF backend instance"""
    global _gguf_backend_instance
    if _gguf_backend_instance is None:
        _gguf_backend_instance = GGUFModelBackend()
    return _gguf_backend_instance
