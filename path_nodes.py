# ComfyUI-QwenVL-Utils
# Path handling nodes

import cv2
from typing import Dict, List, Any, Optional


class MultiplePathsInput:
    """
    Multiple Paths Input
    
    Creates a path batch from multiple paths (images and videos).
    Supports configurable sampling for video files.
    """
    
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
    DESCRIPTION = """
Creates a path batch from multiple paths.
You can set how many inputs the node has,
with the **inputcount** and clicking update.
"""

    IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"}
    VIDEO_EXTENSIONS = {"mp4", "mkv", "mov", "avi", "flv", "wmv", "webm", "m4v"}

    @staticmethod
    def convert_path_to_json(
        file_path: str,
        sample_fps: int = 1,
        max_frames: int = 1,
        use_total_frames: bool = True,
        use_original_fps_as_sample_fps: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Convert a file path to a content JSON object"""
        ext = file_path.split(".")[-1].lower()

        if ext in MultiplePathsInput.IMAGE_EXTENSIONS:
            return {"type": "image", "image": file_path}
        
        elif ext in MultiplePathsInput.VIDEO_EXTENSIONS:
            print(f"[QwenVL-Utils] Processing video: {file_path}")
            
            try:
                vidObj = cv2.VideoCapture(file_path)
                if not vidObj.isOpened():
                    print(f"[QwenVL-Utils] Warning: Could not open video: {file_path}")
                    return None
                
                # Get video properties
                total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
                avg_fps = vidObj.get(cv2.CAP_PROP_FPS)
                width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                vidObj.release()
                
                if total_frames > 0 and avg_fps > 0:
                    duration = total_frames / avg_fps
                    print(f"[QwenVL-Utils] Video info: {total_frames} frames, {avg_fps:.2f} fps, "
                          f"{duration:.2f}s, {width}x{height}")
                else:
                    print(f"[QwenVL-Utils] Warning: Could not get video properties")
                    total_frames = max_frames
                    avg_fps = sample_fps
                
                return {
                    "type": "video",
                    "video": file_path,
                    "fps": avg_fps if use_original_fps_as_sample_fps else sample_fps,
                    "max_frames": total_frames if use_total_frames else max_frames,
                }
                
            except Exception as e:
                print(f"[QwenVL-Utils] Error processing video {file_path}: {e}")
                return None
        else:
            print(f"[QwenVL-Utils] Unsupported file type: {ext}")
            return None

    def combine(self, inputcount: int, **kwargs) -> tuple:
        """Combine multiple paths into a single list"""
        path_list = []
        
        # Filter out path parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith("path_")
        }
        
        for c in range(inputcount):
            path_key = f"path_{c + 1}"
            if path_key not in kwargs:
                continue
                
            path = kwargs[path_key]
            result = self.convert_path_to_json(path, **filtered_kwargs)
            
            if result is not None:
                print(f"[QwenVL-Utils] Added path {c + 1}: {result}")
                path_list.append(result)
        
        print(f"[QwenVL-Utils] Total paths: {len(path_list)}")
        return (path_list,)


# Node mappings for this module
NODE_CLASS_MAPPINGS = {
    "QwenVLUtils_MultiplePathsInput": MultiplePathsInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLUtils_MultiplePathsInput": "Multiple Paths Input",
}
