# ComfyUI-QwenVL-Utils
# Input utility nodes: ImageLoader, VideoLoader, VideoLoaderPath

import hashlib
import os
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.latest import InputImpl

from . import node_helpers


class ImageLoader:
    """
    Load Image Advanced
    
    Loads an image and returns the image, mask, and file path.
    Supports various image formats including animated images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.split(".")[-1].lower() in ["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"]
        ]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            },
        }

    CATEGORY = "QwenVL-Utils/Input"
    RETURN_TYPES = ("IMAGE", "MASK", "PATH")
    RETURN_NAMES = ("image", "mask", "path")
    FUNCTION = "load_image"

    def load_image(self, image: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None
        
        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda x: x * (1 / 255))
            image_frame = i.convert("RGB")

            if len(output_images) == 0:
                w = image_frame.size[0]
                h = image_frame.size[1]

            if image_frame.size[0] != w or image_frame.size[1] != h:
                continue

            image_array = np.array(image_frame).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]
            
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            
            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, image_path)

    @classmethod
    def IS_CHANGED(cls, image: str) -> str:
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image: str):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


class VideoLoader(ComfyNodeABC):
    """
    Load Video Advanced
    
    Loads a video file and returns the video object and file path.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "file": (sorted(files), {"video_upload": True})
            },
        }

    CATEGORY = "QwenVL-Utils/Input"
    RETURN_TYPES = (IO.VIDEO, "PATH")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "load_video"

    def load_video(self, file: str):
        video_path = folder_paths.get_annotated_filepath(file)
        return (InputImpl.VideoFromFile(video_path), video_path)

    @classmethod
    def IS_CHANGED(cls, file: str) -> float:
        video_path = folder_paths.get_annotated_filepath(file)
        # Use modification time instead of hashing for large files
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(cls, file: str):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


class VideoLoaderPath(ComfyNodeABC):
    """
    Load Video Advanced (Path)
    
    Loads a video from a file path string.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"placeholder": "X://insert/path/here.mp4"}),
            },
        }

    CATEGORY = "QwenVL-Utils/Input"
    RETURN_TYPES = (IO.VIDEO, "PATH")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "load_video"

    def load_video(self, file: str):
        video_path = folder_paths.get_annotated_filepath(file)
        return (InputImpl.VideoFromFile(video_path), video_path)

    @classmethod
    def IS_CHANGED(cls, file: str) -> float:
        video_path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(cls, file: str):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


# Node mappings for this module
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
