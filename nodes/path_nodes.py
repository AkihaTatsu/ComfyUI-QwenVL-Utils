# ComfyUI-QwenVL-Utils / nodes / path_nodes.py
# Multiple Paths Input node

import cv2
from typing import Dict, Optional, Any


class MultiplePathsInput:
    IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"}
    VIDEO_EXTENSIONS = {"mp4", "mkv", "mov", "avi", "flv", "wmv", "webm", "m4v"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "path_1": ("PATH",),
            },
            "optional": {
                "sample_fps": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "max_frames": ("INT", {"default": 2, "min": 2, "max": (1 << 63) - 1, "step": 1}),
                "use_total_frames": ("BOOLEAN", {"default": True}),
                "use_original_fps_as_sample_fps": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "combine"
    CATEGORY = "QwenVL-Utils/Input"
    DESCRIPTION = (
        "Creates a path batch from multiple paths.\n"
        "Set how many inputs the node has with **inputcount** and click update."
    )

    @staticmethod
    def convert_path_to_json(
        file_path: str, sample_fps: int = 1, max_frames: int = 1,
        use_total_frames: bool = True, use_original_fps_as_sample_fps: bool = True,
    ) -> Optional[Dict[str, Any]]:
        ext = file_path.rsplit(".", 1)[-1].lower()

        if ext in MultiplePathsInput.IMAGE_EXTENSIONS:
            return {"type": "image", "image": file_path}

        if ext in MultiplePathsInput.VIDEO_EXTENSIONS:
            print(f"[QwenVL-Utils] Processing video: {file_path}")
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    print(f"[QwenVL-Utils] Could not open video: {file_path}")
                    return None
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if total <= 0 or fps <= 0:
                    total, fps = max_frames, sample_fps
                return {
                    "type": "video", "video": file_path,
                    "fps": fps if use_original_fps_as_sample_fps else sample_fps,
                    "max_frames": total if use_total_frames else max_frames,
                }
            except Exception as exc:
                print(f"[QwenVL-Utils] Video error {file_path}: {exc}")
                return None

        print(f"[QwenVL-Utils] Unsupported file type: {ext}")
        return None

    def combine(self, inputcount: int, **kwargs) -> tuple:
        filtered = {k: v for k, v in kwargs.items() if not k.startswith("path_")}
        path_list = []
        for c in range(inputcount):
            key = f"path_{c + 1}"
            if key not in kwargs:
                continue
            result = self.convert_path_to_json(kwargs[key], **filtered)
            if result is not None:
                path_list.append(result)
        print(f"[QwenVL-Utils] Total paths: {len(path_list)}")
        return (path_list,)


NODE_CLASS_MAPPINGS = {
    "QwenVLUtils_MultiplePathsInput": MultiplePathsInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLUtils_MultiplePathsInput": "Multiple Paths Input",
}
