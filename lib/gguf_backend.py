# ComfyUI-QwenVL-Utils / lib / gguf_backend.py
# GGUF model backend via llama-cpp-python

import gc
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Any

import torch

try:
    import comfy.utils
    import comfy.model_management
    _COMFY = True
except ImportError:
    _COMFY = False

from tqdm.auto import tqdm

from .settings import GGUF_VL_CATALOG, SYSTEM_PROMPTS
from .model_utils import get_gguf_base_dir, safe_dirname, download_gguf_file, filter_kwargs_for_callable
from .media import tensor_to_base64_png, sample_video_frames
from .device import pick_device, clear_memory
import re

os.environ.setdefault("LLAMA_CUBLAS", "1")
os.environ.setdefault("LLAMA_METAL", "1")

# ---------------------------------------------------------------------------
# Global caches
# ---------------------------------------------------------------------------
# Maps (model_path, llama_version) -> architecture string for unsupported archs
_UNSUPPORTED_ARCH_CACHE: dict = {}


# ---------------------------------------------------------------------------
# GGUF header inspection
# ---------------------------------------------------------------------------

def _read_gguf_architecture(path: str) -> Optional[str]:
    """Read 'general.architecture' from a GGUF file header."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            _ver = struct.unpack("<I", f.read(4))[0]
            _n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def _str(fh):
                ln = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(ln).decode("utf-8", errors="replace")

            for _ in range(min(n_kv, 10)):
                key = _str(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if vtype == 8:
                    val = _str(f)
                elif vtype == 4:
                    val = struct.unpack("<I", f.read(4))[0]
                elif vtype == 5:
                    val = struct.unpack("<i", f.read(4))[0]
                elif vtype == 6:
                    val = struct.unpack("<f", f.read(4))[0]
                elif vtype == 7:
                    val = struct.unpack("<?", f.read(1))[0]
                elif vtype == 10:
                    val = struct.unpack("<Q", f.read(8))[0]
                else:
                    break
                if key == "general.architecture":
                    return str(val) if isinstance(val, str) else None
    except Exception:
        pass
    return None


# Byte sizes for fixed-width GGUF value types
_GGUF_FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_gguf_block_count(path: str) -> Optional[int]:
    """Read the layer/block count from a GGUF file header.

    Parses KV metadata to find ``{arch}.block_count``.  This is used by
    the progressive-fallback loader to compute sensible intermediate
    ``n_gpu_layers`` values when the model is too large for full GPU
    offloading.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            _ver = struct.unpack("<I", f.read(4))[0]
            _n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def _str():
                ln = struct.unpack("<Q", f.read(8))[0]
                return f.read(ln).decode("utf-8", errors="replace")

            def _skip(vt):
                """Skip a value of the given GGUF type. Returns False if unknown."""
                if vt == 8:          # string
                    _str()
                elif vt in _GGUF_FIXED_SIZES:
                    f.read(_GGUF_FIXED_SIZES[vt])
                elif vt == 9:        # array
                    et = struct.unpack("<I", f.read(4))[0]
                    cnt = struct.unpack("<Q", f.read(8))[0]
                    if et == 8:
                        for _ in range(cnt):
                            _str()
                    elif et in _GGUF_FIXED_SIZES:
                        f.read(cnt * _GGUF_FIXED_SIZES[et])
                    else:
                        return False
                else:
                    return False
                return True

            arch = None
            for _ in range(min(n_kv, 128)):
                key = _str()
                vt = struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture" and vt == 8:
                    arch = _str()
                    continue
                if arch and key == f"{arch}.block_count" and vt in (4, 5, 10):
                    fmt = {4: "<I", 5: "<i", 10: "<Q"}[vt]
                    return struct.unpack(fmt, f.read(_GGUF_FIXED_SIZES[vt]))[0]
                if not _skip(vt):
                    break
    except Exception:
        pass
    return None


def _check_architecture_supported(model_path: str) -> Tuple[bool, str]:
    """Quick vocab-only probe to see if llama.cpp supports this arch."""
    arch = _read_gguf_architecture(model_path)
    if arch is None:
        return True, "unknown"
    try:
        import llama_cpp.llama_cpp as _lc
        from llama_cpp._utils import suppress_stdout_stderr
        params = _lc.llama_model_default_params()
        params.vocab_only = True
        with suppress_stdout_stderr(disable=False):
            model = _lc.llama_model_load_from_file(model_path.encode("utf-8"), params)
        if model is not None:
            _lc.llama_model_free(model)
            return True, arch
        return False, arch
    except Exception:
        return True, arch


def _patch_llama_model_del():
    """Suppress 'sampler' AttributeError in LlamaModel.__del__."""
    try:
        from llama_cpp._internals import LlamaModel
        orig = getattr(LlamaModel, "__del__", None)
        if orig is None:
            return

        def _safe_del(self):
            try:
                orig(self)
            except Exception:
                try:
                    if hasattr(self, "_exit_stack"):
                        self._exit_stack.close()
                except Exception:
                    pass

        LlamaModel.__del__ = _safe_del
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GGUF catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _GGUFResolved:
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


def _flatten_catalog() -> dict:
    flat: dict = {}
    seen: set = set()
    repos = (GGUF_VL_CATALOG.get("qwenVL_model") or GGUF_VL_CATALOG.get("vl_repos")
             or GGUF_VL_CATALOG.get("repos") or {})
    for repo_key, repo in repos.items():
        if not isinstance(repo, dict):
            continue
        author = repo.get("author") or repo.get("publisher")
        repo_name = repo.get("repo_name") or repo.get("repo_name_override") or repo_key
        repo_id = repo.get("repo_id") or (f"{author}/{repo_name}" if author and repo_name else None)
        alt = repo.get("alt_repo_ids") or []
        defaults = repo.get("defaults") or {}
        mmproj = repo.get("mmproj_file")
        for mf in (repo.get("model_files") or []):
            display = Path(mf).name
            if display in seen:
                display = f"{display} ({repo_key})"
            seen.add(display)
            flat[display] = {**defaults, "author": author, "repo_dirname": repo_name,
                             "repo_id": repo_id, "alt_repo_ids": alt,
                             "filename": mf, "mmproj_filename": mmproj}
    # Legacy format
    for name, entry in (GGUF_VL_CATALOG.get("models") or {}).items():
        if isinstance(entry, dict):
            flat[name] = entry
    return flat


def get_gguf_vl_models() -> List[str]:
    catalog = _flatten_catalog()
    vl = sorted(k for k, e in catalog.items() if (e or {}).get("mmproj_filename"))
    return vl or ["(no GGUF VL models configured)"]


def _resolve(model_name: str) -> _GGUFResolved:
    catalog = _flatten_catalog()
    entry = catalog.get(model_name) or {}
    if not entry:
        wanted = {model_name, f"{model_name}.gguf"}
        for c in catalog.values():
            if c.get("filename") and Path(c["filename"]).name in wanted:
                entry = c
                break
    if not entry:
        known = sorted(catalog.keys())[:10]
        raise ValueError(
            f"[QwenVL-Utils] GGUF model '{model_name}' not in config. "
            f"Available: {', '.join(known)}{'...' if len(catalog) > 10 else ''}"
        )
    if not entry.get("filename"):
        raise ValueError(f"[QwenVL-Utils] GGUF config for '{model_name}' missing 'filename'.")

    def _i(k, d):
        try:
            return int(entry.get(k, d))
        except Exception:
            return d

    repo_id = entry.get("repo_id")
    author = entry.get("author") or entry.get("publisher")
    repo_dn = entry.get("repo_dirname") or (repo_id.split("/")[-1] if isinstance(repo_id, str) and "/" in repo_id else model_name)
    return _GGUFResolved(
        display_name=model_name, repo_id=repo_id,
        alt_repo_ids=[str(x) for x in (entry.get("alt_repo_ids") or []) if x],
        author=str(author) if author else None,
        repo_dirname=safe_dirname(str(repo_dn)),
        model_filename=str(entry["filename"]),
        mmproj_filename=str(entry["mmproj_filename"]) if entry.get("mmproj_filename") else None,
        context_length=_i("context_length", 8192),
        image_max_tokens=_i("image_max_tokens", 4096),
        n_batch=_i("n_batch", 512),
        gpu_layers=_i("gpu_layers", -1),
        top_k=_i("top_k", 0),
        pool_size=_i("pool_size", 4194304),
    )


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Retry plan builder & logging helpers
# ---------------------------------------------------------------------------

def _build_retry_plan(n_gpu_layers: int, n_ctx: int, has_flash_attn: bool,
                      extra_perf_keys: list, block_count: Optional[int],
                      partial_gpu_safe: bool = True,
                      ) -> List[Tuple[str, dict, bool]]:
    """Build a list of *(description, overrides, vision_safe)* for progressive
    fallback.

    Uses **proportional** reductions instead of fixed values.  GPU layers
    are reduced gradually (~10 % per step) rather than halved.

    ``vision_safe`` is ``True`` only when ``n_gpu_layers`` in the resulting
    configuration is either fully-GPU (-1 or >= total layers) or CPU-only (0).
    **Partial GPU offload is not vision-safe**: the llama.cpp multimodal
    context initializer (``_init_mtmd_context``) will raise a native
    breakpoint exception (0x80000003) that cannot be caught by Python, causing
    the entire process to crash.  Steps flagged as not vision-safe will have
    their chat handler suppressed to avoid this.

    When ``partial_gpu_safe`` is ``False`` (e.g. for MoE architectures), ALL
    partial-GPU steps are omitted entirely.  The llama.cpp native code crashes
    (0x80000003) during both ``_init_mtmd_context`` AND ``decode`` when MoE
    layers are split across GPU and CPU.  Since these are native crashes,
    Python ``try/except`` cannot catch them and the entire process terminates.
    Instead, extra full-GPU attempts with more aggressive context reduction
    are inserted before the CPU-only fallback.

    Override semantics:
      concrete value → replace the base kwarg
      ``None``       → remove the base kwarg entirely
    """
    plan: List[Tuple[str, dict, bool]] = []
    accumulated: dict = {}  # tracks cumulative modifications

    # Helper: is a resolved n_gpu_layers value considered "full GPU" or CPU-only?
    # -1  = all layers on GPU  → vision safe
    #  0  = CPU only           → vision safe
    # >0  = partial offload    → NOT vision safe (mtmd crash)
    # We evaluate safety from the base n_gpu_layers when no override is present.
    def _vision_safe(overrides: dict) -> bool:
        layers = overrides.get("n_gpu_layers", n_gpu_layers)
        return layers <= 0   # -1 (full GPU) or 0 (CPU)

    # ── Step 0: original parameters ──
    plan.append(("original parameters", {}, _vision_safe({})))

    # ── Step 1: disable flash_attn ──
    if has_flash_attn:
        accumulated["flash_attn"] = None
        plan.append(("disable flash_attn", dict(accumulated),
                     _vision_safe(accumulated)))

    # ── Step 2: remove extra performance parameters ──
    has_extra = False
    for k in extra_perf_keys:
        if k != "flash_attn":            # already handled
            accumulated[k] = None
            has_extra = True
    if has_extra:
        plan.append(("remove performance extras", dict(accumulated),
                     _vision_safe(accumulated)))

    # ── Steps 3+: proportional context reduction ──
    # Use proportional fractions (never a forced fixed value).
    if n_ctx > 4096:
        ctx_75 = max(2048, (int(n_ctx * 0.75) // 256) * 256)
        if ctx_75 < n_ctx:
            plan.append((f"reduce context to ~75% ({ctx_75})",
                         {**accumulated, "n_ctx": ctx_75},
                         _vision_safe(accumulated)))
        ctx_50 = max(2048, (int(n_ctx * 0.50) // 256) * 256)
        if ctx_50 < ctx_75:
            accumulated["n_ctx"] = ctx_50
            plan.append((f"reduce context to ~50% ({ctx_50})",
                         dict(accumulated),
                         _vision_safe(accumulated)))
    elif n_ctx > 2048:
        # Modest reduction for moderate contexts
        ctx_75 = max(2048, (int(n_ctx * 0.75) // 256) * 256)
        if ctx_75 < n_ctx:
            accumulated["n_ctx"] = ctx_75
            plan.append((f"reduce context to ~75% ({ctx_75})",
                         dict(accumulated),
                         _vision_safe(accumulated)))

    # Snapshot the (possibly-reduced) context for GPU-layer steps
    reduced_ctx = accumulated.get("n_ctx", n_ctx)

    # ── Steps N+: GPU-layer changes ──
    if n_gpu_layers != 0:
        total = block_count or 32
        effective = total if n_gpu_layers < 0 else min(n_gpu_layers, total)

        if partial_gpu_safe:
            # Gradual GPU layer reduction (~10 % per step)
            fractions = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
            seen_layers: set = set()
            for frac in fractions:
                reduced = max(1, int(effective * frac))
                if reduced < effective and reduced not in seen_layers:
                    seen_layers.add(reduced)
                    step = dict(accumulated)
                    step["n_gpu_layers"] = reduced
                    step.setdefault("n_ctx", reduced_ctx)
                    plan.append((
                        f"GPU layers {reduced}/{effective} ({frac:.0%}), "
                        f"ctx={step['n_ctx']} [vision disabled: partial GPU]",
                        step,
                        False,   # partial GPU → NOT vision safe
                    ))
        else:
            # Partial GPU offload is UNSAFE for this architecture (e.g. MoE).
            # The llama.cpp native code crashes (0x80000003 BREAKPOINT) during
            # both _init_mtmd_context AND llama_decode when model layers are
            # split across GPU and CPU.  These are non-recoverable native
            # crashes that kill the entire process — Python cannot catch them.
            #
            # Instead: try more aggressive full-GPU configurations (deeper
            # context/batch cuts) before falling to CPU-only.
            for target_pct in [0.375, 0.25]:
                target_ctx = max(2048, (int(n_ctx * target_pct) // 256) * 256)
                current_min = accumulated.get("n_ctx", n_ctx)
                if target_ctx < current_min:
                    step = dict(accumulated)
                    step["n_ctx"] = target_ctx
                    accumulated["n_ctx"] = target_ctx
                    plan.append((
                        f"full GPU, ctx ~{target_pct:.0%} ({target_ctx}) "
                        f"[partial GPU unsafe: MoE arch]",
                        step,
                        True,   # full GPU → vision safe
                    ))
            # Try minimum viable context (2048) if not already there
            current_min = accumulated.get("n_ctx", n_ctx)
            if current_min > 2048:
                step = dict(accumulated)
                step["n_ctx"] = 2048
                accumulated["n_ctx"] = 2048
                plan.append((
                    "full GPU, minimum ctx (2048) [partial GPU unsafe: MoE arch]",
                    step,
                    True,   # full GPU → vision safe
                ))
            # Try full-GPU with halved n_batch (reduces peak KV-cache memory)
            for batch_frac, batch_label in [(0.5, "half"), (0.25, "quarter")]:
                step = dict(accumulated)
                step["n_batch"] = max(1, int(512 * batch_frac))
                plan.append((
                    f"full GPU, min ctx, {batch_label} batch ({step['n_batch']}) "
                    f"[partial GPU unsafe: MoE arch]",
                    step,
                    True,   # full GPU → vision safe
                ))

        # Final: CPU only — vision is safe on CPU
        # CPU mode uses system RAM (not VRAM), so restore the ORIGINAL
        # context size.  Previous steps reduced n_ctx to save VRAM, but
        # RAM is typically 2–4× larger than VRAM, making those cuts
        # unnecessary (and harmful — a too-small ctx will cause
        # "Prompt exceeds n_ctx" at inference time).
        cpu_step = dict(accumulated)
        cpu_step["n_gpu_layers"] = 0
        cpu_step["n_ctx"] = n_ctx          # original, not reduced
        cpu_step.pop("n_batch", None)      # reset batch to default too
        plan.append((
            f"CPU only (0 GPU layers), ctx={cpu_step['n_ctx']}",
            cpu_step,
            True,   # CPU-only → vision safe
        ))

    return plan


def _log_attempt_params(attempt: int, desc: str, kw: dict,
                        total_attempts: int):
    """Log the parameters being used for a load attempt."""
    display_keys = [
        "model_path", "n_ctx", "n_gpu_layers", "n_batch", "flash_attn",
        "swa_full", "pool_size", "top_k", "chat_handler",
        "image_min_tokens", "image_max_tokens", "use_mmap", "use_mlock",
        "n_threads", "n_threads_batch",
    ]
    lines = [f"[QwenVL-Utils] Attempt {attempt + 1}/{total_attempts}: {desc}"]
    lines.append("[QwenVL-Utils]   Current parameters:")
    for k in display_keys:
        if k in kw:
            v = kw[k]
            if k == "model_path":
                v = Path(v).name
            elif k == "chat_handler":
                v = type(v).__name__ if v else "None"
            lines.append(f"[QwenVL-Utils]     {k:24s} = {v}")
    for k in sorted(kw):
        if k not in display_keys:
            lines.append(f"[QwenVL-Utils]     {k:24s} = {kw[k]}")
    print("\n".join(lines))


def _log_failure_details(attempt: int, desc: str, exc: Exception,
                         kw: dict):
    """Log detailed failure diagnostics after a load attempt."""
    import traceback as _tb
    lines = [
        f"[QwenVL-Utils] ✗ Attempt {attempt + 1} FAILED ({desc})",
        f"[QwenVL-Utils]   Error type : {type(exc).__name__}",
        f"[QwenVL-Utils]   Error msg  : {exc}",
    ]
    # VRAM snapshot
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            lines.append(
                f"[QwenVL-Utils]   VRAM state : "
                f"{allocated:.2f} GB allocated, "
                f"{reserved:.2f} GB reserved, "
                f"{total_mem:.2f} GB total"
            )
        except Exception:
            pass
    # RAM snapshot
    try:
        import psutil
        mem = psutil.virtual_memory()
        lines.append(
            f"[QwenVL-Utils]   RAM state  : "
            f"{mem.used / 1024**3:.2f} GB used / "
            f"{mem.total / 1024**3:.2f} GB total "
            f"({mem.percent}% utilization)"
        )
    except Exception:
        pass
    # Key parameters that were active
    param_summary = []
    for k in ("n_ctx", "n_gpu_layers", "n_batch", "flash_attn"):
        if k in kw:
            param_summary.append(f"{k}={kw[k]}")
    if param_summary:
        lines.append(f"[QwenVL-Utils]   Params     : {', '.join(param_summary)}")
    # Brief traceback (last 3 frames)
    tb = _tb.format_exception(type(exc), exc, exc.__traceback__)
    tb_short = "".join(tb[-3:]) if len(tb) > 3 else "".join(tb)
    lines.append(f"[QwenVL-Utils]   Traceback:\n{tb_short.rstrip()}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class GGUFModelBackend:
    # Maximum number of KV-cache exhaustion retries during inference.
    _KV_RETRY_MAX = 3

    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self._signature = None
        # The n_ctx value that was actually used to construct self.llm.
        # Needed by the KV-cache retry logic to know what to increase.
        self._loaded_n_ctx: int = 0

    def clear(self):
        """Release all QwenVL-Utils GGUF resources (model + chat handler).

        Only affects QwenVL-Utils allocations; other loaded models and
        their RAM / VRAM usage are **not** touched.
        """
        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                pass
            self.llm = None
        if self.chat_handler is not None:
            try:
                del self.chat_handler
            except Exception:
                pass
            self.chat_handler = None
        self._signature = None
        self._loaded_n_ctx = 0
        # Two GC passes to handle cyclic references in C++ bindings
        gc.collect()
        gc.collect()
        # Release PyTorch's unused cached memory only (does NOT affect
        # active tensors belonging to other ComfyUI models)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cleanup_for_retry(self):
        """Thorough cleanup between retry attempts.

        Ensures each retry starts from an identical clean state by
        releasing ALL QwenVL-Utils-related RAM and VRAM.  Does NOT
        interfere with memory used by other ComfyUI components.
        """
        self.clear()
        # Give the CUDA driver a moment to reclaim freed memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ load
    def load_model(self, model_name, device, ctx=None, n_batch=None,
                   gpu_layers=None, image_max_tokens=None, top_k=None, pool_size=None):
        # Check llama-cpp-python is installed
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "[QwenVL-Utils] llama-cpp-python not installed. "
                "Install: pip install llama-cpp-python"
            )

        resolved = _resolve(model_name)
        base_dir = get_gguf_base_dir()
        author_dir = safe_dirname(resolved.author or "")
        repo_dir = safe_dirname(resolved.repo_dirname)
        target_dir = base_dir / author_dir / repo_dir
        model_path = target_dir / Path(resolved.model_filename).name
        mmproj_path = (target_dir / Path(resolved.mmproj_filename).name
                       if resolved.mmproj_filename else None)

        # Get llama version for caching
        try:
            import llama_cpp as _lm
            _ver = getattr(_lm, "__version__", "unknown")
        except Exception:
            _ver = "unknown"
        cache_key = (str(model_path), str(_ver))

        # ── Fast-fail on previously-detected unsupported architecture ──
        cached_arch = _UNSUPPORTED_ARCH_CACHE.get(cache_key)
        if cached_arch:
            raise RuntimeError(
                f"[QwenVL-Utils] Architecture '{cached_arch}' unsupported by "
                f"llama-cpp-python {_ver}. See earlier error for details."
            )

        # Download if needed
        repo_ids = ([resolved.repo_id] if resolved.repo_id else []) + resolved.alt_repo_ids
        if not model_path.exists():
            if not repo_ids:
                raise FileNotFoundError(
                    f"[QwenVL-Utils] GGUF file not found: {model_path}. "
                    "No repo configured for auto-download."
                )
            download_gguf_file(repo_ids, resolved.model_filename, model_path)
        if mmproj_path and not mmproj_path.exists():
            if not repo_ids:
                raise FileNotFoundError(
                    f"[QwenVL-Utils] mmproj file not found: {mmproj_path}."
                )
            download_gguf_file(repo_ids, resolved.mmproj_filename, mmproj_path)

        device_kind = pick_device(device)
        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size
        n_gpu_layers = (int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers) if device_kind == "cuda" else 0
        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens
        has_mmproj = mmproj_path is not None and mmproj_path.exists()

        sig = (str(model_path), str(mmproj_path) if has_mmproj else "",
               n_ctx, n_batch_val, n_gpu_layers, img_max, top_k_val, pool_size_val)
        if self.llm is not None and self._signature == sig:
            return
        self.clear()

        _patch_llama_model_del()

        # ── Architecture support check ──
        supported, arch_name = _check_architecture_supported(str(model_path))
        if not supported:
            _UNSUPPORTED_ARCH_CACHE[cache_key] = arch_name
            raise ValueError(
                f"[QwenVL-Utils] ERROR: Unsupported model architecture '{arch_name}'\n"
                f"Model: {model_path.name} | llama-cpp-python: {_ver}\n"
                f"Your build does not support '{arch_name}'. Update llama-cpp-python:\n"
                f"  pip install --upgrade llama-cpp-python --force-reinstall --no-cache-dir"
            )
        print(f"[QwenVL-Utils] GGUF architecture: {arch_name}")

        # ── Detect MoE architecture ──
        # Partial GPU offload causes non-recoverable native crashes
        # (0x80000003 BREAKPOINT) for MoE architectures in llama.cpp.
        # Both _init_mtmd_context (vision) and llama_decode (text) crash
        # when MoE layers are split across GPU and CPU.
        is_moe_arch = "moe" in arch_name.lower()
        if is_moe_arch:
            print(
                f"[QwenVL-Utils] MoE architecture detected ('{arch_name}'). "
                f"Partial GPU-layer offload will be SKIPPED entirely \u2014 "
                f"llama.cpp crashes (0x80000003) during decode when MoE "
                f"layers are split across GPU and CPU. "
                f"Will try full-GPU configurations first, then CPU-only "
                f"as last resort."
            )

        # ── Find vision chat handler class (looked up once, instantiated per retry) ──
        handler_cls = None
        if has_mmproj:
            for cls_name in ("Qwen3VLChatHandler", "Qwen25VLChatHandler"):
                try:
                    handler_cls = getattr(
                        __import__("llama_cpp.llama_chat_format", fromlist=[cls_name]),
                        cls_name)
                    break
                except (ImportError, AttributeError):
                    continue
            if handler_cls is None:
                raise RuntimeError(
                    "[QwenVL-Utils] No Qwen VL chat handler found in llama-cpp-python. "
                    "Update: pip install --upgrade llama-cpp-python"
                )

        # ── Detect which kwargs Llama.__init__ accepts ──
        import inspect as _insp
        try:
            explicit = {
                p.name for p in _insp.signature(Llama.__init__).parameters.values()
                if p.kind in (_insp.Parameter.POSITIONAL_OR_KEYWORD,
                              _insp.Parameter.KEYWORD_ONLY)
            }
        except Exception:
            explicit = set()

        # ── Base Llama kwargs (constant across retries) ──
        base_llm_kw: dict = {
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
        if has_mmproj and handler_cls:
            base_llm_kw["image_min_tokens"] = 1024
            base_llm_kw["image_max_tokens"] = img_max

        # Extra performance kwargs (only if Llama accepts them)
        extras = {"swa_full": True, "flash_attn": True,
                  "pool_size": pool_size_val, "top_k": top_k_val}
        active_extras: dict = {}
        for ek, ev in extras.items():
            if ek in explicit:
                base_llm_kw[ek] = ev
                active_extras[ek] = ev

        # Strip keys that Llama.__init__ doesn't accept
        if explicit:
            safe_keys = {"model_path", "n_ctx", "n_gpu_layers", "n_batch",
                         "verbose", "use_mmap", "use_mlock", "n_threads",
                         "n_threads_batch", "chat_handler", "chat_format",
                         "flash_attn", "swa_full", "image_min_tokens",
                         "image_max_tokens"}
            for k in list(base_llm_kw):
                if k not in explicit and k not in safe_keys:
                    base_llm_kw.pop(k, None)

        has_flash = "flash_attn" in base_llm_kw and base_llm_kw.get("flash_attn", False)
        extra_perf_keys = [k for k in active_extras if k in base_llm_kw]
        vision = has_mmproj and handler_cls is not None

        print(f"[QwenVL-Utils] Loading GGUF: {model_path.name} "
              f"(device={device_kind}, gpu_layers={n_gpu_layers}, ctx={n_ctx}, "
              f"flash_attn={has_flash}, vision={vision})")

        # ── Build retry plan ──
        block_count = _read_gguf_block_count(str(model_path)) if n_gpu_layers != 0 else None
        retry_plan = _build_retry_plan(
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            has_flash_attn=has_flash,
            extra_perf_keys=extra_perf_keys,
            block_count=block_count,
            partial_gpu_safe=not is_moe_arch,
        )

        # ── Execute retry plan ──
        last_err = None
        total_attempts = len(retry_plan)
        # Track when model weights fail to load at a given GPU-layer
        # count.  Reducing n_ctx / n_batch only shrinks the KV cache,
        # NOT the model weights — so further attempts at the *same*
        # n_gpu_layers value are futile and should be skipped.
        _weights_failed_at: Optional[int] = None   # n_gpu_layers that failed

        for attempt, (desc, overrides, vision_safe) in enumerate(retry_plan):
            # ── Skip futile retries when model weights don't fit ──
            resolved_layers = overrides.get("n_gpu_layers", n_gpu_layers)
            if (_weights_failed_at is not None
                    and resolved_layers == _weights_failed_at):
                print(
                    f"[QwenVL-Utils] Skipping '{desc}' — model weights "
                    f"don't fit in VRAM at n_gpu_layers="
                    f"{resolved_layers}.  Reducing ctx/batch only "
                    f"shrinks KV cache, not model weights."
                )
                continue

            # ── Ensure a clean state before every attempt ──
            if attempt > 0:
                print(f"\n[QwenVL-Utils] ── Retry {attempt}/{total_attempts - 1}: "
                      f"{desc} ──")
                self._cleanup_for_retry()

            # ── Apply overrides to a fresh copy of base kwargs ──
            kw = dict(base_llm_kw)
            for ok, ov in overrides.items():
                if ov is None:
                    kw.pop(ok, None)
                else:
                    kw[ok] = ov

            # ── Create chat handler fresh for this attempt ──
            # Recreated each time so that the retry starts at the exact
            # same memory state (mmproj is released in _cleanup_for_retry).
            #
            # IMPORTANT: partial GPU-layer offload is NOT vision-safe.
            # The llama.cpp multimodal context initializer (_init_mtmd_context)
            # issues a native breakpoint trap (Windows 0x80000003) when the
            # model uses partial GPU offload, killing the entire process — this
            # cannot be caught by Python.  We therefore suppress the chat
            # handler for any step flagged vision_safe=False, allowing the
            # model to load as text-only for that configuration.
            if has_mmproj and handler_cls and vision_safe:
                try:
                    mmproj_kw = filter_kwargs_for_callable(
                        getattr(handler_cls, "__init__", handler_cls),
                        {"clip_model_path": str(mmproj_path),
                         "image_max_tokens": img_max,
                         "force_reasoning": False, "verbose": False},
                    )
                    self.chat_handler = handler_cls(**mmproj_kw)
                    kw["chat_handler"] = self.chat_handler
                except Exception as handler_exc:
                    print(f"[QwenVL-Utils] Warning: chat handler creation "
                          f"failed: {handler_exc}")
                    self.chat_handler = None
                    kw.pop("chat_handler", None)
            elif has_mmproj and handler_cls and not vision_safe:
                print(
                    f"[QwenVL-Utils] Vision (mmproj) suppressed for this attempt: "
                    f"partial GPU-layer offload causes a non-recoverable native "
                    f"crash in _init_mtmd_context.  "
                    f"Model will load as text-only for this configuration."
                )
                self.chat_handler = None
                kw.pop("chat_handler", None)

            try:
                self.llm = Llama(**kw)
                if attempt > 0:
                    print(f"[QwenVL-Utils] Successfully loaded with: {desc}")
                last_err = None
                break
            except (ValueError, RuntimeError, OSError) as exc:
                last_err = exc
                _log_attempt_params(attempt, desc, kw, total_attempts)
                _log_failure_details(attempt, desc, exc, kw)
                # Unsupported architecture → no point retrying
                if "unknown model architecture" in str(exc).lower():
                    break
                # Model weights don't fit → skip remaining steps at
                # the same n_gpu_layers (ctx/batch changes won't help).
                err_str = str(exc).lower()
                if "failed to load model from file" in err_str:
                    _weights_failed_at = kw.get("n_gpu_layers", n_gpu_layers)
                    print(
                        f"[QwenVL-Utils] Model weights failed to load at "
                        f"n_gpu_layers={_weights_failed_at}. Will skip "
                        f"remaining attempts at this GPU-layer count."
                    )
                self.llm = None
                self.chat_handler = None

        if last_err is not None:
            # Final cleanup before raising
            self._cleanup_for_retry()
            raise ValueError(
                f"[QwenVL-Utils] Failed to load GGUF '{model_path.name}' "
                f"after {total_attempts} attempt(s) "
                f"(arch={arch_name}, llama-cpp={_ver}): {last_err}"
            ) from last_err

        # Record the actual n_ctx for KV-cache retry logic
        self._loaded_n_ctx = kw.get("n_ctx", n_ctx)
        self._signature = sig

    # -------------------------------------------------------------- generate
    def _invoke(self, system_prompt, user_prompt, images_b64,
                max_tokens, temperature, top_p, repetition_penalty, seed,
                min_p=0.0, top_k_sampling=0):
        if images_b64:
            content = [{"type": "text", "text": user_prompt}]
            for img in images_b64:
                if img:
                    content.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img}"}})
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}]
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]

        start = time.perf_counter()

        # Temperature handling: match HF backend behavior
        # Very low temperature -> greedy (temperature=0 in llama.cpp)
        temp = float(temperature)
        if temp < 0.01:
            temp = 0.0  # greedy decoding

        # Sampling parameters passed explicitly to override llama-cpp-python
        # defaults (min_p=0.05, top_k=40) which can cause premature EOS.
        # Default values (min_p=0.0, top_k=0) match HF backend behavior.
        common = dict(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=temp,
            top_p=float(top_p),
            top_k=int(top_k_sampling),
            min_p=float(min_p),
            repeat_penalty=float(repetition_penalty),
            seed=int(seed),
        )

        if _COMFY:
            pbar = comfy.utils.ProgressBar(max_tokens)
            tqdm_bar = tqdm(total=int(max_tokens), desc="Generating", unit="token", leave=True)
            tokens, parts, interrupted = 0, [], False
            finish_reason = None
            for chunk in self.llm.create_chat_completion(**common, stream=True):
                if comfy.model_management.processing_interrupted():
                    interrupted = True
                    break
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta", {})
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
                c = delta.get("content", "")
                if c:
                    parts.append(c)
                    tokens += 1
                    pbar.update_absolute(tokens, max_tokens)
                    tqdm_bar.update(1)
            if not interrupted:
                pbar.update_absolute(tokens, tokens)
            remaining = tokens - tqdm_bar.n
            if remaining > 0:
                tqdm_bar.update(remaining)
            tqdm_bar.close()
            elapsed = max(time.perf_counter() - start, 1e-6)
            if tokens:
                reason_str = f", finish={finish_reason}" if finish_reason else ""
                print(f"[QwenVL-Utils] {tokens} tokens in {elapsed:.2f}s "
                      f"({tokens/elapsed:.1f} tok/s{reason_str})"
                      + (" (interrupted)" if interrupted else ""))
            content_str = "".join(parts)
            if interrupted:
                comfy.model_management.throw_exception_if_processing_interrupted()
        else:
            result = self.llm.create_chat_completion(**common)
            elapsed = max(time.perf_counter() - start, 1e-6)
            usage = result.get("usage") or {}
            ct = usage.get("completion_tokens")
            if isinstance(ct, int) and ct > 0:
                print(f"[QwenVL-Utils] {ct} tokens in {elapsed:.2f}s ({ct/elapsed:.1f} tok/s)")
            content_str = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")

        raw = str(content_str or "")

        # ── Match HF backend behavior: minimal cleanup ──
        # The HF backend simply decodes with skip_special_tokens=True.
        # The aggressive output_cleaner was stripping <think> blocks
        # (removing ALL content inside them), causing models that put
        # their detailed response inside <think> tags to return only
        # the brief final answer (e.g. just "yes").
        #
        # Instead: strip special tokens and <think>/</ think> TAGS
        # (but KEEP content inside them), same as HF does.
        cleaned = raw

        # 1. Remove chat/special tokens (like HF skip_special_tokens)
        cleaned = re.sub(
            r'<\|?im_(start|end)\|?>|<im_(start|end)>|<\|endoftext\|>',
            '', cleaned, flags=re.IGNORECASE
        ).strip()

        # 2. Remove <think> / </think> TAGS but KEEP content inside
        cleaned = re.sub(r'</?think[^>]*>', '', cleaned, flags=re.IGNORECASE).strip()

        # 3. Strip role prefix on first line (e.g. "assistant\n")
        cleaned = re.sub(r'^\s*assistant\s*\n', '', cleaned, flags=re.IGNORECASE).strip()

        if raw and not cleaned:
            # Safety: if cleaning removed everything, log and return raw
            print(f"[QwenVL-Utils] WARNING: output cleaner removed all content! "
                  f"Raw output ({len(raw)} chars): {raw[:300]}")
            cleaned = raw

        # Debug: show raw vs cleaned when they differ significantly
        if len(raw) > 50 and len(cleaned) < len(raw) * 0.5:
            print(f"[QwenVL-Utils] DEBUG raw output ({len(raw)} chars): {raw[:500]}")
            print(f"[QwenVL-Utils] DEBUG cleaned output ({len(cleaned)} chars): {cleaned[:500]}")

        return cleaned

    # ------------------------------------------------------------------- run
    def run(self, model_name, preset_prompt, custom_prompt, image, video,
            frame_count, max_tokens, temperature, top_p, repetition_penalty,
            seed, keep_model_loaded, device,
            ctx=None, n_batch=None, gpu_layers=None,
            image_max_tokens=None, top_k=None, pool_size=None,
            min_p=0.0, top_k_sampling=0) -> Tuple[str]:

        try:
            import llama_cpp as _lm
            _ver = getattr(_lm, "__version__", "unknown")
        except Exception:
            _ver = "unknown"
        if _COMFY:
            comfy.model_management.throw_exception_if_processing_interrupted()

        torch.manual_seed(int(seed))

        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()

        images_b64 = []
        if image is not None:
            b64 = tensor_to_base64_png(image)
            if b64:
                images_b64.append(b64)
        if video is not None:
            for f in sample_video_frames(video, int(frame_count)):
                b64 = tensor_to_base64_png(f)
                if b64:
                    images_b64.append(b64)

        try:
            self.load_model(model_name, device, ctx, n_batch, gpu_layers,
                            image_max_tokens, top_k, pool_size)

            if images_b64 and self.chat_handler is None:
                print("[QwenVL-Utils] Warning: images provided but no mmproj; images ignored")

            # ── Inference with KV-cache exhaustion retry ──
            # If llama_decode fails with "No KV slot available", the context
            # window is too small for the prompt + generated tokens.  We
            # reload the model with a proportionally larger n_ctx and retry.
            kv_retry = 0
            current_ctx_override = ctx  # user-provided or None
            while True:
                try:
                    text = self._invoke(
                        system_prompt="You are a helpful vision-language assistant.",
                        user_prompt=prompt,
                        images_b64=images_b64 if self.chat_handler else [],
                        max_tokens=max_tokens, temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty, seed=seed,
                        min_p=min_p, top_k_sampling=top_k_sampling,
                    )
                    return (text,)
                except (RuntimeError, ValueError) as kv_exc:
                    err_msg = str(kv_exc).lower()
                    is_kv_error = (
                        "no kv slot" in err_msg
                        or "kv_cache" in err_msg
                        or ("llama_decode failed" in err_msg
                            and "context" in err_msg)
                        or "prompt exceeds n_ctx" in err_msg
                    )
                    if not is_kv_error or kv_retry >= self._KV_RETRY_MAX:
                        raise

                    kv_retry += 1
                    old_ctx = self._loaded_n_ctx
                    # Proportional increase: +50% each retry, aligned to 256
                    new_ctx = ((int(old_ctx * 1.5) + 255) // 256) * 256

                    print(
                        f"\n[QwenVL-Utils] KV cache exhausted during inference "
                        f"(ctx={old_ctx}).\n"
                        f"[QwenVL-Utils]   Error: {kv_exc}\n"
                        f"[QwenVL-Utils]   KV retry {kv_retry}/{self._KV_RETRY_MAX}: "
                        f"reloading model with n_ctx {old_ctx} → {new_ctx}"
                    )
                    if torch.cuda.is_available():
                        try:
                            alloc = torch.cuda.memory_allocated() / 1024**3
                            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            print(f"[QwenVL-Utils]   VRAM: {alloc:.2f}/{total:.2f} GB")
                        except Exception:
                            pass

                    # Force reload with the larger context
                    current_ctx_override = new_ctx
                    self.clear()  # full cleanup
                    self.load_model(
                        model_name, device,
                        ctx=current_ctx_override,
                        n_batch=n_batch,
                        gpu_layers=gpu_layers,
                        image_max_tokens=image_max_tokens,
                        top_k=top_k,
                        pool_size=pool_size,
                    )
                    print(
                        f"[QwenVL-Utils]   Model reloaded with n_ctx="
                        f"{self._loaded_n_ctx}. Retrying inference..."
                    )
        except Exception:
            raise
        finally:
            if not keep_model_loaded:
                self.clear()


_gguf_backend_instance = None

def get_gguf_backend() -> GGUFModelBackend:
    global _gguf_backend_instance
    if _gguf_backend_instance is None:
        _gguf_backend_instance = GGUFModelBackend()
    return _gguf_backend_instance
