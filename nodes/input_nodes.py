# ComfyUI-QwenVL-Utils / nodes / input_nodes.py
# ImageLoader, VideoLoader, VideoLoaderPath

import hashlib
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.latest import InputImpl

from .. import node_helpers


class ImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.split(".")[-1].lower() in {"jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"}
        ]
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

    CATEGORY = "QwenVL-Utils/Input"
    RETURN_TYPES = ("IMAGE", "MASK", "PATH")
    RETURN_NAMES = ("image", "mask", "path")
    FUNCTION = "load_image"

    def load_image(self, image: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)
        output_images, output_masks = [], []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == "I":
                i = i.point(lambda x: x * (1 / 255))
            frame = i.convert("RGB")
            if w is None:
                w, h = frame.size
            if frame.size != (w, h):
                continue
            arr = np.array(frame).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(arr)[None,])
            if "A" in i.getbands():
                mask = 1.0 - torch.from_numpy(np.array(i.getchannel("A")).astype(np.float32) / 255.0)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in ("MPO",):
            out_img = torch.cat(output_images, dim=0)
            out_mask = torch.cat(output_masks, dim=0)
        else:
            out_img, out_mask = output_images[0], output_masks[0]
        return (out_img, out_mask, image_path)

    @classmethod
    def IS_CHANGED(cls, image):
        path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


class VideoLoader(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(
            [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))],
            ["video"],
        )
        return {"required": {"file": (sorted(files), {"video_upload": True})}}

    CATEGORY = "QwenVL-Utils/Input"
    RETURN_TYPES = (IO.VIDEO, "PATH")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "load_video"

    def load_video(self, file):
        path = folder_paths.get_annotated_filepath(file)
        return (InputImpl.VideoFromFile(path), path)

    @classmethod
    def IS_CHANGED(cls, file):
        return os.path.getmtime(folder_paths.get_annotated_filepath(file))

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


class VideoLoaderPath(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file": ("STRING", {"placeholder": "X://insert/path/here.mp4"})}}

    CATEGORY = "QwenVL-Utils/Input"
    RETURN_TYPES = (IO.VIDEO, "PATH")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "load_video"

    def load_video(self, file):
        path = folder_paths.get_annotated_filepath(file)
        return (InputImpl.VideoFromFile(path), path)

    @classmethod
    def IS_CHANGED(cls, file):
        return os.path.getmtime(folder_paths.get_annotated_filepath(file))

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


NODE_CLASS_MAPPINGS = {
    "QwenVLUtils_ImageLoader": ImageLoader,
    "QwenVLUtils_VideoLoader": VideoLoader,
    "QwenVLUtils_VideoLoaderPath": VideoLoaderPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLUtils_ImageLoader": "Load Image Advanced",
    "QwenVLUtils_VideoLoader": "Load Video Advanced",
    "QwenVLUtils_VideoLoaderPath": "Load Video Advanced (Path)",
}
