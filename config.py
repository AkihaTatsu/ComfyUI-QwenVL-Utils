# ComfyUI-QwenVL-Utils
# Configuration and constants for QwenVL nodes
#
# This integration supports Qwen-VL series models including Qwen3-VL and Qwen2.5-VL
# for advanced multimodal AI with text generation, image understanding, and video analysis.

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "hf_models.json"
SYSTEM_PROMPTS_PATH = NODE_DIR / "system_prompts.json"
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"
TOOLTIPS_PATH = NODE_DIR / "tooltips.json"

# Model storage
HF_VL_MODELS: Dict[str, dict] = {}
HF_TEXT_MODELS: Dict[str, dict] = {}
HF_ALL_MODELS: Dict[str, dict] = {}
GGUF_VL_CATALOG: Dict[str, Any] = {}
SYSTEM_PROMPTS: Dict[str, str] = {}
PRESET_PROMPTS: List[str] = []
TOOLTIPS: Dict[str, str] = {}


class Quantization(str, Enum):
    """Quantization options for model loading"""
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
        
        available = ", ".join([f"'{item.value}'" for item in cls])
        error_msg = (
            f"\n" + "="*70 + "\n"
            f"[QwenVL-Utils] ERROR: Invalid quantization value: '{value}'\n"
            "="*70 + "\n"
            f"Available options: {available}\n"
            "\n"
            "Quantization guide:\n"
            "  - '4-bit' (Q4): Lowest VRAM, fastest, slight quality loss\n"
            "  - '8-bit' (Q8): Balanced VRAM and quality\n"
            "  - 'None (FP16)': Best quality, highest VRAM\n"
            "="*70
        )
        raise ValueError(error_msg)


class ModelType(str, Enum):
    """Model type enumeration"""
    HF = "huggingface"
    GGUF = "gguf"


def load_config_from_qwenvl():
    """Try to load config from ComfyUI-QwenVL if it exists"""
    global HF_VL_MODELS, HF_TEXT_MODELS, HF_ALL_MODELS, SYSTEM_PROMPTS, PRESET_PROMPTS, GGUF_VL_CATALOG, TOOLTIPS
    
    # Try to find ComfyUI-QwenVL installation
    qwenvl_paths = [
        NODE_DIR.parent / "ComfyUI-QwenVL",
        NODE_DIR.parent / "comfyui-qwenvl",
    ]
    
    qwenvl_dir = None
    for path in qwenvl_paths:
        if path.exists():
            qwenvl_dir = path
            break
    
    if qwenvl_dir:
        # Load HF models config
        hf_config = qwenvl_dir / "hf_models.json"
        if hf_config.exists():
            try:
                with open(hf_config, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
                HF_VL_MODELS = data.get("hf_vl_models") or {}
                HF_TEXT_MODELS = data.get("hf_text_models") or {}
                PRESET_PROMPTS = data.get("_preset_prompts") or []
                SYSTEM_PROMPTS = data.get("_system_prompts") or {}
            except Exception as exc:
                print(f"[QwenVL-Utils] Failed to load HF models from QwenVL: {exc}")
        
        # Load GGUF models config
        gguf_config = qwenvl_dir / "gguf_models.json"
        if gguf_config.exists():
            try:
                with open(gguf_config, "r", encoding="utf-8") as fh:
                    GGUF_VL_CATALOG = json.load(fh) or {}
            except Exception as exc:
                print(f"[QwenVL-Utils] Failed to load GGUF models from QwenVL: {exc}")
        
        # Load system prompts
        sys_prompts = qwenvl_dir / "AILab_System_Prompts.json"
        if sys_prompts.exists():
            try:
                with open(sys_prompts, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
                qwenvl_prompts = data.get("qwenvl") or {}
                preset_override = data.get("_preset_prompts") or []
                if qwenvl_prompts:
                    SYSTEM_PROMPTS = qwenvl_prompts
                if preset_override:
                    PRESET_PROMPTS = preset_override
            except Exception as exc:
                print(f"[QwenVL-Utils] Failed to load system prompts from QwenVL: {exc}")
    
    # Always load local config for anything not yet loaded
    _load_local_config()
    
    # Combine all models
    HF_ALL_MODELS = dict(HF_VL_MODELS)
    HF_ALL_MODELS.update(HF_TEXT_MODELS)


def _load_local_config():
    """Load configuration from local files"""
    global HF_VL_MODELS, HF_TEXT_MODELS, SYSTEM_PROMPTS, PRESET_PROMPTS, GGUF_VL_CATALOG, TOOLTIPS
    
    # Load HF models
    if not HF_VL_MODELS and CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            HF_VL_MODELS = data.get("hf_vl_models") or {}
            HF_TEXT_MODELS = data.get("hf_text_models") or {}
        except Exception as exc:
            print(f"[QwenVL-Utils] Local HF config load failed: {exc}")
    
    # Load GGUF models
    if not GGUF_VL_CATALOG and GGUF_CONFIG_PATH.exists():
        try:
            with open(GGUF_CONFIG_PATH, "r", encoding="utf-8") as fh:
                GGUF_VL_CATALOG = json.load(fh) or {}
        except Exception as exc:
            print(f"[QwenVL-Utils] GGUF config load failed: {exc}")
    
    # Load system prompts
    if (not SYSTEM_PROMPTS or not PRESET_PROMPTS) and SYSTEM_PROMPTS_PATH.exists():
        try:
            with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            if not SYSTEM_PROMPTS:
                SYSTEM_PROMPTS = data.get("qwenvl") or {}
            if not PRESET_PROMPTS:
                PRESET_PROMPTS = data.get("_preset_prompts") or []
        except Exception as exc:
            print(f"[QwenVL-Utils] System prompts load failed: {exc}")
    
    # Load tooltips
    if not TOOLTIPS and TOOLTIPS_PATH.exists():
        try:
            with open(TOOLTIPS_PATH, "r", encoding="utf-8") as fh:
                TOOLTIPS = json.load(fh) or {}
        except Exception as exc:
            print(f"[QwenVL-Utils] Tooltips load failed: {exc}")


# Initialize configuration on module load
load_config_from_qwenvl()