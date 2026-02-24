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

class GGUFModelBackend:
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self._signature = None

    def clear(self):
        self.llm = self.chat_handler = None
        self._signature = None
        clear_memory()

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

        # ── Chat handler for vision ──
        self.chat_handler = None
        if has_mmproj:
            handler_cls = None
            for cls_name in ("Qwen3VLChatHandler", "Qwen25VLChatHandler"):
                try:
                    handler_cls = getattr(__import__("llama_cpp.llama_chat_format", fromlist=[cls_name]),
                                          cls_name)
                    break
                except (ImportError, AttributeError):
                    continue
            if handler_cls is None:
                raise RuntimeError(
                    "[QwenVL-Utils] No Qwen VL chat handler found in llama-cpp-python. "
                    "Update: pip install --upgrade llama-cpp-python"
                )
            mmproj_kw = filter_kwargs_for_callable(
                getattr(handler_cls, "__init__", handler_cls),
                {"clip_model_path": str(mmproj_path), "image_max_tokens": img_max,
                 "force_reasoning": False, "verbose": False},
            )
            self.chat_handler = handler_cls(**mmproj_kw)

        # ── Llama constructor kwargs ──
        llm_kw: dict = {
            "model_path": str(model_path),
            "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers, "n_batch": n_batch_val,
            "verbose": False, "use_mmap": True, "use_mlock": False,
            "n_threads": max(1, os.cpu_count() // 2),
            "n_threads_batch": max(1, os.cpu_count() // 2),
        }
        if has_mmproj and self.chat_handler:
            llm_kw["chat_handler"] = self.chat_handler
            llm_kw["image_min_tokens"] = 1024
            llm_kw["image_max_tokens"] = img_max

        # Only add extra kwargs if Llama.__init__ explicitly accepts them
        import inspect as _insp
        try:
            explicit = {
                p.name for p in _insp.signature(Llama.__init__).parameters.values()
                if p.kind in (_insp.Parameter.POSITIONAL_OR_KEYWORD, _insp.Parameter.KEYWORD_ONLY)
            }
        except Exception:
            explicit = set()

        extras = {"swa_full": True, "flash_attn": True, "pool_size": pool_size_val, "top_k": top_k_val}
        for ek, ev in extras.items():
            if ek in explicit:
                llm_kw[ek] = ev

        # Strip unknown explicit keys
        if explicit:
            safe_keys = {"model_path", "n_ctx", "n_gpu_layers", "n_batch", "verbose",
                         "use_mmap", "use_mlock", "n_threads", "n_threads_batch",
                         "chat_handler", "chat_format", "flash_attn", "swa_full"}
            for k in list(llm_kw):
                if k not in explicit and k not in safe_keys:
                    llm_kw.pop(k, None)

        flash = llm_kw.get("flash_attn", False)
        vision = has_mmproj and self.chat_handler is not None
        print(f"[QwenVL-Utils] Loading GGUF: {model_path.name} "
              f"(device={device_kind}, gpu_layers={n_gpu_layers}, ctx={n_ctx}, "
              f"flash_attn={flash}, vision={vision})")

        # ── Progressive fallback loading ──
        # When the model + vision mmproj don't fit in VRAM, the previous
        # strategy jumped straight from "all GPU layers" to "CPU only".
        # We now insert intermediate steps that reduce n_gpu_layers so
        # the user gets *partial* GPU acceleration instead of none.
        fallbacks = [("full", {})]
        if llm_kw.get("flash_attn"):
            fallbacks.append(("no flash_attn", {"flash_attn": False}))
        perf_keys = [k for k in extras if k in llm_kw]
        no_perf = {k: None for k in perf_keys}
        if perf_keys:
            fallbacks.append(("no perf extras", dict(no_perf)))
        if n_ctx > 2048:
            fallbacks.append((f"ctx={n_ctx // 2}", {**no_perf, "n_ctx": n_ctx // 2}))
        if n_gpu_layers != 0:
            # Determine the total layer count from the GGUF header so
            # we can try sensible partial-offload values.
            block_count = _read_gguf_block_count(str(model_path))
            total_layers = block_count or 32  # safe default
            effective = total_layers if n_gpu_layers < 0 else min(n_gpu_layers, total_layers)
            seen_layers: set = set()
            for frac_label, frac in [("half", 0.5), ("quarter", 0.25)]:
                reduced = max(1, int(effective * frac))
                if reduced < effective and reduced not in seen_layers:
                    seen_layers.add(reduced)
                    fallbacks.append((
                        f"{frac_label} GPU layers ({reduced})",
                        {**no_perf, "n_gpu_layers": reduced,
                         "n_ctx": min(n_ctx, 4096)},
                    ))
            fallbacks.append(("CPU only", {**no_perf,
                                           "n_gpu_layers": 0, "n_ctx": min(n_ctx, 4096)}))

        last_err = None
        for idx, (desc, mods) in enumerate(fallbacks):
            kw = dict(llm_kw)
            for mk, mv in mods.items():
                if mv is None:
                    kw.pop(mk, None)
                else:
                    kw[mk] = mv
            if idx > 0:
                print(f"[QwenVL-Utils] Retry {idx}/{len(fallbacks)-1}: {desc}")
            try:
                self.llm = Llama(**kw)
                if idx > 0:
                    print(f"[QwenVL-Utils] Loaded with fallback: {desc}")
                last_err = None
                break
            except (ValueError, RuntimeError, OSError) as exc:
                last_err = exc
                print(f"[QwenVL-Utils] Load failed ({desc}): {exc}")
                if "unknown model architecture" in str(exc).lower():
                    break
                self.llm = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if last_err is not None:
            raise ValueError(
                f"[QwenVL-Utils] Failed to load GGUF '{model_path.name}' "
                f"(arch={arch_name}, llama-cpp={_ver}): {last_err}"
            ) from last_err

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
            if not interrupted:
                pbar.update_absolute(tokens, tokens)
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

            text = self._invoke(
                system_prompt="You are a helpful vision-language assistant.",
                user_prompt=prompt,
                images_b64=images_b64 if self.chat_handler else [],
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty, seed=seed,
                min_p=min_p, top_k_sampling=top_k_sampling,
            )
            return (text,)
        except Exception:
            raise
        finally:
            if not keep_model_loaded:
                self.llm = self.chat_handler = None
                self._signature = None
                clear_memory()


_gguf_backend_instance = None

def get_gguf_backend() -> GGUFModelBackend:
    global _gguf_backend_instance
    if _gguf_backend_instance is None:
        _gguf_backend_instance = GGUFModelBackend()
    return _gguf_backend_instance
