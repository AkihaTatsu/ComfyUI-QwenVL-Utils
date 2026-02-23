# ComfyUI-QwenVL-Utils / lib / media.py
# Image and video tensor conversion utilities

import base64
import io
from typing import Optional, List

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Optional[Image.Image]:
    if tensor is None:
        return None
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def tensor_to_base64_png(tensor: torch.Tensor) -> Optional[str]:
    if tensor is None:
        return None
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    pil_img = Image.fromarray(array, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def sample_video_frames(video: torch.Tensor, frame_count: int) -> List[torch.Tensor]:
    if video is None:
        return []
    if video.ndim != 4:
        return [video]
    total = int(video.shape[0])
    frame_count = max(int(frame_count), 1)
    if total <= frame_count:
        return [video[i] for i in range(total)]
    idx = np.linspace(0, total - 1, frame_count, dtype=int)
    return [video[i] for i in idx]
