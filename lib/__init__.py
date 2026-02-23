# ComfyUI-QwenVL-Utils / lib
# Functional libraries for model loading and inference

from .settings import (
    HF_VL_MODELS,
    HF_TEXT_MODELS,
    HF_ALL_MODELS,
    GGUF_VL_CATALOG,
    SYSTEM_PROMPTS,
    PRESET_PROMPTS,
    TOOLTIPS,
    Quantization,
    ModelType,
)
from .attention import ATTENTION_MODES, resolve_attention_mode, SageAttentionContext
from .device import (
    get_device_info,
    get_device_options,
    normalize_device_choice,
    pick_device,
    is_bf16_supported,
    get_optimal_dtype,
    clear_memory,
    enforce_memory_limits,
)
from .media import (
    tensor_to_pil,
    tensor_to_base64_png,
    sample_video_frames,
)
from .model_utils import (
    get_model_save_path,
    get_gguf_base_dir,
    ensure_hf_model,
    safe_dirname,
    download_gguf_file,
    optimize_model_for_inference,
    try_torch_compile,
    filter_kwargs_for_callable,
)
from .hf_backend import HFModelBackend, get_hf_backend
from .gguf_backend import GGUFModelBackend, get_gguf_backend, get_gguf_vl_models
from .output_cleaner import clean_model_output, OutputCleanConfig
