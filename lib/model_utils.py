# ComfyUI-QwenVL-Utils / lib / model_utils.py
# Shared model utilities: download, paths, optimization, helpers

import inspect
import os
from pathlib import Path
from typing import Optional, List, Dict, Callable

import torch
from huggingface_hub import snapshot_download, hf_hub_download
import folder_paths

from .settings import HF_ALL_MODELS, GGUF_VL_CATALOG


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_model_save_path() -> Path:
    llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    if llm_paths:
        return Path(llm_paths[0]) / "Qwen-VL"
    return Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"


def get_gguf_base_dir() -> Path:
    base_dir_value = GGUF_VL_CATALOG.get("base_dir", "LLM/GGUF")
    base_dir = Path(base_dir_value)
    if base_dir.is_absolute():
        return base_dir
    return Path(folder_paths.models_dir) / base_dir


def safe_dirname(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    return "".join(ch for ch in value if ch.isalnum() or ch in "._- ").strip() or "unknown"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def ensure_hf_model(model_name: str) -> str:
    """Download HF model if not present; return local path."""
    info = HF_ALL_MODELS.get(model_name)
    if not info:
        known = sorted(HF_ALL_MODELS.keys())[:10]
        raise ValueError(
            f"[QwenVL-Utils] HF model '{model_name}' not in config. "
            f"Available: {', '.join(known)}{'...' if len(HF_ALL_MODELS) > 10 else ''}"
        )

    repo_id = info["repo_id"]
    target = get_model_save_path() / repo_id.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)

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


def download_gguf_file(repo_ids: List[str], filename: str, target_path: Path):
    if target_path.exists():
        print(f"[QwenVL-Utils] Cached: {target_path}")
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc = None
    for repo_id in repo_ids:
        print(f"[QwenVL-Utils] Downloading {filename} from {repo_id}...")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type="model",
                local_dir=str(target_path.parent), local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                downloaded_path.replace(target_path)
            if target_path.exists():
                print(f"[QwenVL-Utils] Download complete: {target_path}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"[QwenVL-Utils] Download from {repo_id} failed: {exc}")
    else:
        raise FileNotFoundError(
            f"[QwenVL-Utils] Failed to download {filename}. "
            f"Last error: {last_exc}. Repos tried: {repo_ids}"
        ) from last_exc

    if not target_path.exists():
        raise FileNotFoundError(f"[QwenVL-Utils] File missing after download: {target_path}")


# ---------------------------------------------------------------------------
# Model optimization
# ---------------------------------------------------------------------------

def optimize_model_for_inference(model):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    for param in model.parameters():
        param.requires_grad = False
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return model


def try_torch_compile(model, device: str):
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return model
    try:
        major, minor = torch.__version__.split(".")[:2]
        if int(major) < 2:
            print("[QwenVL-Utils] torch.compile requires PyTorch 2.0+")
            return model
    except Exception:
        return model

    compile_mode = (
        "max-autotune" if os.environ.get("QWENVL_MAX_COMPILE", "0") == "1"
        else "reduce-overhead"
    )
    print(f"[QwenVL-Utils] Applying torch.compile (mode={compile_mode})...")

    for fullgraph in (True, False):
        try:
            compiled = torch.compile(model, mode=compile_mode, fullgraph=fullgraph, dynamic=not fullgraph)
            fg_label = "fullgraph" if fullgraph else "partial"
            print(f"[QwenVL-Utils] torch.compile OK ({fg_label}, mode={compile_mode})")
            return compiled
        except Exception:
            continue

    try:
        compiled = torch.compile(model, mode="default")
        print("[QwenVL-Utils] torch.compile OK (default mode)")
        return compiled
    except Exception as exc:
        print(f"[QwenVL-Utils] torch.compile failed: {exc}")
        return model


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def filter_kwargs_for_callable(fn: Callable, kwargs: Dict) -> Dict:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)
    allowed = {
        p.name for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in allowed}
