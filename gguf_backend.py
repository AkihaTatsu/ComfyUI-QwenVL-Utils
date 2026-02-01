# ComfyUI-QwenVL-Utils
# GGUF model backend for QwenVL (via llama-cpp-python)
# Optimized for maximum inference speed

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Any

import torch

from .config import GGUF_VL_CATALOG, SYSTEM_PROMPTS
from .utils import (
    get_gguf_base_dir,
    safe_dirname,
    download_gguf_file,
    tensor_to_base64_png,
    sample_video_frames,
    pick_device,
    clear_memory,
    filter_kwargs_for_callable,
)
from .output_cleaner import clean_model_output, OutputCleanConfig

# Performance: Set llama.cpp environment variables before import
os.environ.setdefault("LLAMA_CUBLAS", "1")  # Enable CUDA acceleration
os.environ.setdefault("LLAMA_METAL", "1")   # Enable Metal on macOS


@dataclass(frozen=True)
class GGUFModelResolved:
    """Resolved GGUF model configuration"""
    display_name: str
    repo_id: Optional[str]
    alt_repo_ids: List[str]
    author: Optional[str]
    repo_dirname: str
    model_filename: str
    mmproj_filename: Optional[str]
    context_length: int
    image_max_tokens: int
    n_batch: int
    gpu_layers: int
    top_k: int
    pool_size: int


def _flatten_gguf_catalog() -> dict:
    """Flatten the GGUF catalog into a single dict of models"""
    flattened = {}
    seen_display_names = set()
    
    repos = (
        GGUF_VL_CATALOG.get("qwenVL_model") or 
        GGUF_VL_CATALOG.get("vl_repos") or 
        GGUF_VL_CATALOG.get("repos") or 
        {}
    )
    
    for repo_key, repo in repos.items():
        if not isinstance(repo, dict):
            continue
        
        author = repo.get("author") or repo.get("publisher")
        repo_name = repo.get("repo_name") or repo.get("repo_name_override") or repo_key
        repo_id = repo.get("repo_id") or (f"{author}/{repo_name}" if author and repo_name else None)
        alt_repo_ids = repo.get("alt_repo_ids") or []
        
        defaults = repo.get("defaults") or {}
        mmproj_file = repo.get("mmproj_file")
        model_files = repo.get("model_files") or []
        
        for model_file in model_files:
            display = Path(model_file).name
            if display in seen_display_names:
                display = f"{display} ({repo_key})"
            seen_display_names.add(display)
            
            flattened[display] = {
                **defaults,
                "author": author,
                "repo_dirname": repo_name,
                "repo_id": repo_id,
                "alt_repo_ids": alt_repo_ids,
                "filename": model_file,
                "mmproj_filename": mmproj_file,
            }
    
    # Also include legacy "models" format
    legacy_models = GGUF_VL_CATALOG.get("models") or {}
    for name, entry in legacy_models.items():
        if isinstance(entry, dict):
            flattened[name] = entry
    
    return flattened


def get_gguf_vl_models() -> List[str]:
    """Get list of available GGUF VL models"""
    all_models = _flatten_gguf_catalog()
    return sorted([
        key for key, entry in all_models.items() 
        if (entry or {}).get("mmproj_filename")
    ]) or ["(no GGUF VL models configured)"]


def _resolve_model_entry(model_name: str) -> GGUFModelResolved:
    """Resolve a model name to its full configuration"""
    all_models = _flatten_gguf_catalog()
    entry = all_models.get(model_name) or {}
    
    if not entry:
        # Try to find by filename
        wanted = {model_name, f"{model_name}.gguf"}
        for candidate in all_models.values():
            filename = candidate.get("filename")
            if filename and Path(filename).name in wanted:
                entry = candidate
                break
    
    repo_id = entry.get("repo_id")
    alt_repo_ids = entry.get("alt_repo_ids") or []
    
    author = entry.get("author") or entry.get("publisher")
    repo_dirname = entry.get("repo_dirname") or (
        repo_id.split("/")[-1] if isinstance(repo_id, str) and "/" in repo_id else model_name
    )
    
    model_filename = entry.get("filename")
    mmproj_filename = entry.get("mmproj_filename")
    
    if not model_filename:
        raise ValueError(f"[QwenVL-Utils] GGUF config missing 'filename' for: {model_name}")
    
    def _int(name: str, default: int) -> int:
        value = entry.get(name, default)
        try:
            return int(value)
        except Exception:
            return default
    
    return GGUFModelResolved(
        display_name=model_name,
        repo_id=repo_id,
        alt_repo_ids=[str(x) for x in alt_repo_ids if x],
        author=str(author) if author else None,
        repo_dirname=safe_dirname(str(repo_dirname)),
        model_filename=str(model_filename),
        mmproj_filename=str(mmproj_filename) if mmproj_filename else None,
        context_length=_int("context_length", 8192),
        image_max_tokens=_int("image_max_tokens", 4096),
        n_batch=_int("n_batch", 512),
        gpu_layers=_int("gpu_layers", -1),
        top_k=_int("top_k", 0),
        pool_size=_int("pool_size", 4194304),
    )


class GGUFModelBackend:
    """GGUF model backend for QwenVL inference via llama-cpp-python - Optimized for speed"""
    
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
    
    def clear(self):
        """Clear loaded model and free memory"""
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        clear_memory()
    
    def _check_backend(self):
        """Check if llama-cpp-python is available"""
        try:
            from llama_cpp import Llama  # noqa: F401
        except ImportError as exc:
            error_msg = (
                "\n" + "="*70 + "\n"
                "[QwenVL-Utils] ERROR: llama-cpp-python not found\n"
                "="*70 + "\n"
                "GGUF models require llama-cpp-python with vision support.\n\n"
                "Installation options:\n"
                "  1. With CUDA (NVIDIA GPU) - Fastest:\n"
                "     pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n\n"
                "  2. CPU only:\n"
                "     pip install llama-cpp-python\n\n"
                "  3. Metal (Apple Silicon):\n"
                "     CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python\n\n"
                "For detailed instructions, see:\n"
                "  https://github.com/1038lab/ComfyUI-QwenVL/blob/main/docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md\n"
                "="*70
            )
            raise RuntimeError(error_msg) from exc
    
    def load_model(
        self,
        model_name: str,
        device: str,
        ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        gpu_layers: Optional[int] = None,
        image_max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        pool_size: Optional[int] = None,
    ):
        """Load GGUF model"""
        self._check_backend()
        
        resolved = _resolve_model_entry(model_name)
        base_dir = get_gguf_base_dir()
        
        author_dir = safe_dirname(resolved.author or "")
        repo_dir = safe_dirname(resolved.repo_dirname)
        target_dir = base_dir / author_dir / repo_dir
        
        model_path = target_dir / Path(resolved.model_filename).name
        mmproj_path = target_dir / Path(resolved.mmproj_filename).name if resolved.mmproj_filename else None
        
        # Prepare repo IDs for download
        repo_ids = []
        if resolved.repo_id:
            repo_ids.append(resolved.repo_id)
        repo_ids.extend(resolved.alt_repo_ids)
        
        # Download model if not present
        if not model_path.exists():
            if not repo_ids:
                error_msg = (
                    f"\n" + "="*70 + "\n"
                    f"[QwenVL-Utils] ERROR: GGUF model file not found\n"
                    "="*70 + "\n"
                    f"Model: {model_name}\n"
                    f"Expected path: {model_path}\n"
                    f"Filename: {resolved.model_filename}\n\n"
                    "No repository configured for automatic download.\n\n"
                    "Manual download required:\n"
                    "  1. Find the GGUF model file online\n"
                    "  2. Download the .gguf file\n"
                    f"  3. Place it in: {target_dir}\n"
                    f"  4. Ensure filename matches: {resolved.model_filename}\n"
                    "="*70
                )
                raise FileNotFoundError(error_msg)
            download_gguf_file(repo_ids, resolved.model_filename, model_path)
        
        # Download mmproj if needed
        if mmproj_path is not None and not mmproj_path.exists():
            if not repo_ids:
                error_msg = (
                    f"\n" + "="*70 + "\n"
                    f"[QwenVL-Utils] ERROR: GGUF mmproj file not found\n"
                    "="*70 + "\n"
                    f"Model: {model_name}\n"
                    f"Expected path: {mmproj_path}\n"
                    f"Filename: {resolved.mmproj_filename}\n\n"
                    "Vision models require both .gguf and mmproj files.\n"
                    "No repository configured for automatic download.\n\n"
                    "Manual download required:\n"
                    f"  1. Download mmproj file from model repository\n"
                    f"  2. Place it in: {target_dir}\n"
                    f"  3. Ensure filename matches: {resolved.mmproj_filename}\n"
                    "="*70
                )
                raise FileNotFoundError(error_msg)
            download_gguf_file(repo_ids, resolved.mmproj_filename, mmproj_path)
        
        device_kind = pick_device(device)
        
        # Resolve parameters
        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size
        
        n_gpu_layers = 0
        if device_kind == "cuda":
            n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
        
        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens
        
        has_mmproj = mmproj_path is not None and mmproj_path.exists()
        
        # Check signature for caching
        signature = (
            str(model_path),
            str(mmproj_path) if has_mmproj else "",
            n_ctx,
            n_batch_val,
            n_gpu_layers,
            img_max,
            top_k_val,
            pool_size_val,
        )
        
        if self.llm is not None and self.current_signature == signature:
            return
        
        self.clear()
        
        from llama_cpp import Llama
        
        # Load chat handler for vision
        self.chat_handler = None
        if has_mmproj:
            handler_cls = None
            try:
                from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                handler_cls = Qwen3VLChatHandler
            except ImportError:
                try:
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                    handler_cls = Qwen25VLChatHandler
                except ImportError:
                    error_msg = (
                        "\n" + "="*70 + "\n"
                        "[QwenVL-Utils] ERROR: Qwen VL chat handler not found\n"
                        "="*70 + "\n"
                        "Your llama-cpp-python installation does not support Qwen VL models.\n\n"
                        "Solutions:\n"
                        "  1. Update llama-cpp-python to latest version:\n"
                        "     pip install --upgrade llama-cpp-python\n\n"
                        "  2. Install multimodal-capable build (recommended):\n"
                        "     pip uninstall llama-cpp-python\n"
                        "     pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n\n"
                        "  3. Check for compatible version:\n"
                        "     pip install 'llama-cpp-python>=0.2.70'\n\n"
                        "Note: Qwen3VLChatHandler or Qwen25VLChatHandler required for vision support.\n"
                        "="*70
                    )
                    raise RuntimeError(error_msg)
            
            mmproj_kwargs = {
                "clip_model_path": str(mmproj_path),
                "image_max_tokens": img_max,
                "force_reasoning": False,
                "verbose": False,
            }
            mmproj_kwargs = filter_kwargs_for_callable(
                getattr(handler_cls, "__init__", handler_cls), 
                mmproj_kwargs
            )
            if "image_max_tokens" not in mmproj_kwargs:
                print("[QwenVL-Utils] Warning: chat handler does not support image_max_tokens")
            
            self.chat_handler = handler_cls(**mmproj_kwargs)
        
        # Prepare Llama kwargs with performance optimizations
        llm_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "swa_full": True,
            "verbose": False,
            "pool_size": pool_size_val,
            "top_k": top_k_val,
            # Performance optimizations
            "use_mmap": True,              # Memory-map model for faster loading
            "use_mlock": False,            # Don't lock in RAM (allows swapping if needed)
            "flash_attn": True,            # Use Flash Attention if available
            "n_threads": max(1, os.cpu_count() // 2),  # Parallel CPU threads
            "n_threads_batch": max(1, os.cpu_count() // 2),  # Parallel batch threads
        }
        
        if has_mmproj and self.chat_handler is not None:
            llm_kwargs["chat_handler"] = self.chat_handler
            llm_kwargs["image_min_tokens"] = 1024
            llm_kwargs["image_max_tokens"] = img_max
        
        print(f"[QwenVL-Utils] Loading GGUF: {model_path.name} (device={device_kind}, gpu_layers={n_gpu_layers}, ctx={n_ctx}, flash_attn=True)")
        
        llm_kwargs_filtered = filter_kwargs_for_callable(
            getattr(Llama, "__init__", Llama), 
            llm_kwargs
        )
        
        if has_mmproj and self.chat_handler is not None and "chat_handler" not in llm_kwargs_filtered:
            print("[QwenVL-Utils] Warning: Llama() does not accept chat_handler; images will be ignored")
        
        if device_kind == "cuda" and n_gpu_layers == 0:
            print("[QwenVL-Utils] Warning: device=cuda but n_gpu_layers=0; model runs on CPU")
        
        self.llm = Llama(**llm_kwargs_filtered)
        self.current_signature = signature
    
    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
    ) -> str:
        """Invoke the model for generation"""
        if images_b64:
            content = [{"type": "text", "text": user_prompt}]
            for img in images_b64:
                if not img:
                    continue
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{img}"}
                })
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        
        start = time.perf_counter()
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repeat_penalty=float(repetition_penalty),
            seed=int(seed),
            stop=["<|im_end|>", "<|im_start|>"],
        )
        elapsed = max(time.perf_counter() - start, 1e-6)
        
        # Log performance
        usage = result.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            tok_s = completion_tokens / elapsed
            if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                print(f"[QwenVL-Utils] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, "
                      f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")
            else:
                print(f"[QwenVL-Utils] Tokens: completion={completion_tokens}, "
                      f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")
        
        content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        cleaned = clean_model_output(str(content or ""), OutputCleanConfig(mode="text"))
        return cleaned.strip()
    
    def run(
        self,
        model_name: str,
        preset_prompt: str,
        custom_prompt: str,
        image: Optional[Any],
        video: Optional[Any],
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        device: str,
        ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        gpu_layers: Optional[int] = None,
        image_max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        pool_size: Optional[int] = None,
    ) -> Tuple[str]:
        """Run inference with the model"""
        torch.manual_seed(int(seed))
        
        # Resolve prompt
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        
        # Prepare images
        images_b64 = []
        if image is not None:
            img = tensor_to_base64_png(image)
            if img:
                images_b64.append(img)
        
        if video is not None:
            for frame in sample_video_frames(video, int(frame_count)):
                img = tensor_to_base64_png(frame)
                if img:
                    images_b64.append(img)
        
        try:
            self.load_model(
                model_name=model_name,
                device=device,
                ctx=ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                image_max_tokens=image_max_tokens,
                top_k=top_k,
                pool_size=pool_size,
            )
            
            if images_b64 and self.chat_handler is None:
                print("[QwenVL-Utils] Warning: images provided but model has no mmproj; images ignored")
            
            text = self._invoke(
                system_prompt=(
                    "You are a helpful vision-language assistant. "
                    "Answer directly with the final answer only. No <think> and no reasoning."
                ),
                user_prompt=prompt,
                images_b64=images_b64 if self.chat_handler is not None else [],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


# Global instance for reuse
_gguf_backend_instance = None


def get_gguf_backend() -> GGUFModelBackend:
    """Get or create the GGUF backend instance"""
    global _gguf_backend_instance
    if _gguf_backend_instance is None:
        _gguf_backend_instance = GGUFModelBackend()
    return _gguf_backend_instance
