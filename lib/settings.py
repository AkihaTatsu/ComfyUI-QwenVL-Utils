# ComfyUI-QwenVL-Utils / lib / settings.py
# Configuration loader: reads JSON config files and exposes globals

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any

# Paths
_PKG_DIR = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PKG_DIR / "config"

# Global config storage
HF_VL_MODELS: Dict[str, dict] = {}
HF_TEXT_MODELS: Dict[str, dict] = {}
HF_ALL_MODELS: Dict[str, dict] = {}
GGUF_VL_CATALOG: Dict[str, Any] = {}
SYSTEM_PROMPTS: Dict[str, str] = {}
PRESET_PROMPTS: List[str] = []
TOOLTIPS: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Quantization(str, Enum):
    Q4 = "4-bit (VRAM-friendly)"
    Q8 = "8-bit (Balanced)"
    FP16 = "None (FP16)"

    @classmethod
    def get_values(cls) -> List[str]:
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value: str) -> "Quantization":
        for item in cls:
            if item.value == value:
                return item
        available = ", ".join(f"'{i.value}'" for i in cls)
        raise ValueError(
            f"[QwenVL-Utils] Invalid quantization '{value}'. Options: {available}"
        )


class ModelType(str, Enum):
    HF = "huggingface"
    GGUF = "gguf"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    """Load a JSON file, returning {} on failure."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as exc:
        print(f"[QwenVL-Utils] Failed to load {path.name}: {exc}")
        return {}


def _load_local_config():
    """Load configuration from local JSON files."""
    global HF_VL_MODELS, HF_TEXT_MODELS, GGUF_VL_CATALOG, SYSTEM_PROMPTS, PRESET_PROMPTS, TOOLTIPS

    # HF models
    if not HF_VL_MODELS:
        data = _load_json(_CONFIG_DIR / "hf_models.json")
        HF_VL_MODELS = data.get("hf_vl_models") or {}
        HF_TEXT_MODELS = data.get("hf_text_models") or {}

    # GGUF models
    if not GGUF_VL_CATALOG:
        GGUF_VL_CATALOG = _load_json(_CONFIG_DIR / "gguf_models.json")

    # System prompts
    if not SYSTEM_PROMPTS or not PRESET_PROMPTS:
        data = _load_json(_CONFIG_DIR / "system_prompts.json")
        if not SYSTEM_PROMPTS:
            SYSTEM_PROMPTS = data.get("qwenvl") or {}
        if not PRESET_PROMPTS:
            PRESET_PROMPTS = data.get("_preset_prompts") or []

    # Tooltips
    if not TOOLTIPS:
        TOOLTIPS = _load_json(_CONFIG_DIR / "tooltips.json")


def load_config():
    """Load configuration, trying sibling ComfyUI-QwenVL first, then local."""
    global HF_VL_MODELS, HF_TEXT_MODELS, HF_ALL_MODELS
    global GGUF_VL_CATALOG, SYSTEM_PROMPTS, PRESET_PROMPTS, TOOLTIPS

    # Try sibling ComfyUI-QwenVL installation
    for name in ("ComfyUI-QwenVL", "comfyui-qwenvl"):
        qwenvl_dir = _PKG_DIR.parent / name
        if not qwenvl_dir.exists():
            continue

        hf_cfg = qwenvl_dir / "hf_models.json"
        if hf_cfg.exists():
            data = _load_json(hf_cfg)
            HF_VL_MODELS = data.get("hf_vl_models") or {}
            HF_TEXT_MODELS = data.get("hf_text_models") or {}
            PRESET_PROMPTS = data.get("_preset_prompts") or []
            SYSTEM_PROMPTS = data.get("_system_prompts") or {}

        gguf_cfg = qwenvl_dir / "gguf_models.json"
        if gguf_cfg.exists():
            GGUF_VL_CATALOG = _load_json(gguf_cfg)

        sys_cfg = qwenvl_dir / "AILab_System_Prompts.json"
        if sys_cfg.exists():
            data = _load_json(sys_cfg)
            sp = data.get("qwenvl") or {}
            pp = data.get("_preset_prompts") or []
            if sp:
                SYSTEM_PROMPTS = sp
            if pp:
                PRESET_PROMPTS = pp
        break

    # Fill in anything not loaded from sibling
    _load_local_config()

    # Combine all HF models
    HF_ALL_MODELS = dict(HF_VL_MODELS)
    HF_ALL_MODELS.update(HF_TEXT_MODELS)


# Auto-load on import
load_config()
