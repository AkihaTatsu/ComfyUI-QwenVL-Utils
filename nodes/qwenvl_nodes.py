# ComfyUI-QwenVL-Utils / nodes / qwenvl_nodes.py
# QwenVL (Basic) and QwenVL (Advanced) ComfyUI nodes

import torch
from typing import Optional, List, Tuple

from ..lib.settings import (
    HF_VL_MODELS, GGUF_VL_CATALOG, PRESET_PROMPTS, SYSTEM_PROMPTS,
    Quantization, ModelType, TOOLTIPS,
)
from ..lib.attention import ATTENTION_MODES
from ..lib.device import get_device_options
from ..lib.hf_backend import HFModelBackend, get_hf_backend
from ..lib.gguf_backend import GGUFModelBackend, get_gguf_backend, get_gguf_vl_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_model_names() -> Tuple[List[str], str]:
    hf = list(HF_VL_MODELS.keys())
    gguf = get_gguf_vl_models()
    gguf_marked = [f"[GGUF] {m}" for m in gguf if not m.startswith("(")]
    all_models = hf + gguf_marked
    if not all_models:
        all_models = ["Qwen3-VL-4B-Instruct"]
    default = hf[0] if hf else (gguf_marked[0] if gguf_marked else all_models[0])
    return all_models, default


def _parse_choice(name: str) -> Tuple[str, ModelType]:
    if name.startswith("[GGUF] "):
        return name[7:], ModelType.GGUF
    return name, ModelType.HF


def _default_prompt():
    prompts = PRESET_PROMPTS or ["🖼️ Detailed Description"]
    preferred = "🖼️ Detailed Description"
    return prompts, (preferred if preferred in prompts else prompts[0])


# ---------------------------------------------------------------------------
# QwenVL (Basic)
# ---------------------------------------------------------------------------

class QwenVLBasic:
    _hf_backend: Optional[HFModelBackend] = None
    _gguf_backend: Optional[GGUFModelBackend] = None

    @classmethod
    def INPUT_TYPES(cls):
        all_models, default_model = _all_model_names()
        prompts, default_prompt = _default_prompt()
        return {
            "required": {
                "model_name": (all_models, {
                    "default": default_model,
                    "tooltip": TOOLTIPS.get("model_name", ""),
                }),
                "quantization": (Quantization.get_values(), {
                    "default": Quantization.FP16.value,
                    "tooltip": TOOLTIPS.get("quantization", ""),
                }),
                "attention_mode": (ATTENTION_MODES, {
                    "default": "auto",
                    "tooltip": TOOLTIPS.get("attention_mode", ""),
                }),
                "preset_prompt": (prompts, {
                    "default": default_prompt,
                    "tooltip": TOOLTIPS.get("preset_prompt", ""),
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": TOOLTIPS.get("custom_prompt", ""),
                }),
                "max_tokens": ("INT", {
                    "default": 512, "min": 64, "max": 256000,
                    "tooltip": TOOLTIPS.get("max_tokens", ""),
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS.get("keep_model_loaded", ""),
                }),
                "seed": ("INT", {
                    "default": 1, "min": 1, "max": 2**32 - 1,
                    "tooltip": TOOLTIPS.get("seed", ""),
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "source_path": ("PATH",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "QwenVL-Utils"

    def process(self, model_name, quantization, attention_mode, preset_prompt,
                custom_prompt, max_tokens, keep_model_loaded, seed,
                image=None, video=None, source_path=None, unique_id=None):
        actual, mtype = _parse_choice(model_name)
        if mtype == ModelType.GGUF:
            return get_gguf_backend().run(
                model_name=actual, preset_prompt=preset_prompt,
                custom_prompt=custom_prompt, image=image, video=video,
                frame_count=16, max_tokens=max_tokens, temperature=0.6,
                top_p=0.9, repetition_penalty=1.2, seed=seed,
                keep_model_loaded=keep_model_loaded, device="auto",
                ctx=None, n_batch=None, gpu_layers=None,
                image_max_tokens=None, top_k=None, pool_size=None,
                min_p=0.0, top_k_sampling=0,
            )
        return get_hf_backend().run(
            model_name=actual, quantization=quantization,
            preset_prompt=preset_prompt, custom_prompt=custom_prompt,
            image=image, video=video, frame_count=16, max_tokens=max_tokens,
            temperature=0.6, top_p=0.9, num_beams=1, repetition_penalty=1.2,
            seed=seed, keep_model_loaded=keep_model_loaded,
            attention_mode=attention_mode, use_torch_compile=False,
            device="auto", unique_id=unique_id,
        )


# ---------------------------------------------------------------------------
# QwenVL (Advanced)
# ---------------------------------------------------------------------------

class QwenVLAdvanced:
    _hf_backend: Optional[HFModelBackend] = None
    _gguf_backend: Optional[GGUFModelBackend] = None

    @classmethod
    def INPUT_TYPES(cls):
        all_models, default_model = _all_model_names()
        prompts, default_prompt = _default_prompt()
        device_options = get_device_options()
        return {
            "required": {
                "model_name": (all_models, {
                    "default": default_model,
                    "tooltip": TOOLTIPS.get("model_name", ""),
                }),
                "quantization": (Quantization.get_values(), {
                    "default": Quantization.FP16.value,
                    "tooltip": TOOLTIPS.get("quantization", ""),
                }),
                "attention_mode": (ATTENTION_MODES, {
                    "default": "auto",
                    "tooltip": TOOLTIPS.get("attention_mode", ""),
                }),
                "use_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS.get("use_torch_compile", ""),
                }),
                "device": (device_options, {
                    "default": "auto",
                    "tooltip": TOOLTIPS.get("device", ""),
                }),
                "preset_prompt": (prompts, {
                    "default": default_prompt,
                    "tooltip": TOOLTIPS.get("preset_prompt", ""),
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": TOOLTIPS.get("custom_prompt", ""),
                }),
                "max_tokens": ("INT", {
                    "default": 512, "min": 64, "max": 256000,
                    "tooltip": TOOLTIPS.get("max_tokens", ""),
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": TOOLTIPS.get("temperature", ""),
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": TOOLTIPS.get("top_p", ""),
                }),
                "num_beams": ("INT", {
                    "default": 1, "min": 1, "max": 8,
                    "tooltip": TOOLTIPS.get("num_beams", ""),
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.2, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": TOOLTIPS.get("repetition_penalty", ""),
                }),
                "frame_count": ("INT", {
                    "default": 16, "min": 1, "max": 64,
                    "tooltip": TOOLTIPS.get("frame_count", ""),
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS.get("keep_model_loaded", ""),
                }),
                "seed": ("INT", {
                    "default": 1, "min": 1, "max": 2**32 - 1,
                    "tooltip": TOOLTIPS.get("seed", ""),
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "source_path": ("PATH",),
                "min_pixels": ("INT", {
                    "default": 256 * 28 * 28, "min": 4 * 28 * 28,
                    "max": 16384 * 28 * 28, "step": 28 * 28,
                    "tooltip": TOOLTIPS.get("min_pixels", ""),
                }),
                "max_pixels": ("INT", {
                    "default": 1280 * 28 * 28, "min": 4 * 28 * 28,
                    "max": 16384 * 28 * 28, "step": 28 * 28,
                    "tooltip": TOOLTIPS.get("max_pixels", ""),
                }),
                "ctx": ("INT", {
                    "default": 8192, "min": 1024, "max": 262144, "step": 512,
                    "tooltip": TOOLTIPS.get("ctx", ""),
                }),
                "n_batch": ("INT", {
                    "default": 512, "min": 64, "max": 32768, "step": 64,
                    "tooltip": TOOLTIPS.get("n_batch", ""),
                }),
                "gpu_layers": ("INT", {
                    "default": -1, "min": -1, "max": 200,
                    "tooltip": TOOLTIPS.get("gpu_layers", ""),
                }),
                "image_max_tokens": ("INT", {
                    "default": 4096, "min": 256, "max": 1024000, "step": 256,
                    "tooltip": TOOLTIPS.get("image_max_tokens", ""),
                }),
                "top_k": ("INT", {
                    "default": 0, "min": 0, "max": 32768,
                    "tooltip": TOOLTIPS.get("top_k", ""),
                }),
                "min_p": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": TOOLTIPS.get("min_p", ""),
                }),
                "top_k_sampling": ("INT", {
                    "default": 0, "min": 0, "max": 1000,
                    "tooltip": TOOLTIPS.get("top_k_sampling", ""),
                }),
                "pool_size": ("INT", {
                    "default": 4194304, "min": 1048576, "max": 10485760, "step": 524288,
                    "tooltip": TOOLTIPS.get("pool_size", ""),
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "QwenVL-Utils"

    def process(self, model_name, quantization, attention_mode, use_torch_compile,
                device, preset_prompt, custom_prompt, max_tokens, temperature,
                top_p, num_beams, repetition_penalty, frame_count,
                keep_model_loaded, seed,
                image=None, video=None, source_path=None,
                min_pixels=None, max_pixels=None,
                ctx=None, n_batch=None, gpu_layers=None,
                image_max_tokens=None, top_k=None, min_p=None,
                top_k_sampling=None, pool_size=None,
                unique_id=None):
        actual, mtype = _parse_choice(model_name)
        if mtype == ModelType.GGUF:
            return get_gguf_backend().run(
                model_name=actual, preset_prompt=preset_prompt,
                custom_prompt=custom_prompt, image=image, video=video,
                frame_count=frame_count, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty, seed=seed,
                keep_model_loaded=keep_model_loaded, device=device,
                ctx=ctx, n_batch=n_batch, gpu_layers=gpu_layers,
                image_max_tokens=image_max_tokens, top_k=top_k, pool_size=pool_size,
                min_p=min_p if min_p is not None else 0.0,
                top_k_sampling=top_k_sampling if top_k_sampling is not None else 0,
            )
        return get_hf_backend().run(
            model_name=actual, quantization=quantization,
            preset_prompt=preset_prompt, custom_prompt=custom_prompt,
            image=image, video=video, frame_count=frame_count,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            num_beams=num_beams, repetition_penalty=repetition_penalty,
            seed=seed, keep_model_loaded=keep_model_loaded,
            attention_mode=attention_mode, use_torch_compile=use_torch_compile,
            device=device, min_pixels=min_pixels, max_pixels=max_pixels,
            unique_id=unique_id,
        )


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "QwenVL_Basic": QwenVLBasic,
    "QwenVL_Advanced": QwenVLAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL_Basic": "QwenVL (Basic)",
    "QwenVL_Advanced": "QwenVL (Advanced)",
}
