# ComfyUI-QwenVL-Utils

A comprehensive and modular QwenVL integration for ComfyUI, providing advanced vision-language capabilities with support for both HuggingFace Transformers and GGUF models. This extension consolidates features from multiple QwenVL implementations while introducing enhanced error handling, attention backend optimization, and a clean, maintainable codebase.

## Credits & Acknowledgments

This project builds upon and consolidates features from multiple excellent QwenVL implementations:

### Original Implementations
* **ComfyUI-QwenVL** by [1038lab](https://github.com/1038lab/ComfyUI-QwenVL)
  - GGUF backend integration
  - System prompt templates
  - Advanced parameter controls
  - Comprehensive model support
  
* **ComfyUI_Qwen2-VL-Instruct** by [IuvenisSapiens](https://github.com/IuvenisSapiens/ComfyUI_Qwen2-VL-Instruct)
  - Initial ComfyUI integration
  - Multi-image and video support
  - Path node implementation
  - Clean workflow design

## Supported Models

### HuggingFace Vision-Language Models

| Model | Size | Features | VRAM (FP16) | VRAM (8-bit) | VRAM (4-bit) |
|-------|------|----------|-------------|--------------|--------------|
| Qwen3-VL-2B-Instruct | 2B | General VL | ~4GB | ~2.5GB | ~1.5GB |
| Qwen3-VL-2B-Thinking | 2B | CoT reasoning | ~4GB | ~2.5GB | ~1.5GB |
| Qwen3-VL-4B-Instruct | 4B | Balanced | ~6GB | ~3.5GB | ~2GB |
| Qwen3-VL-4B-Thinking | 4B | CoT reasoning | ~6GB | ~3.5GB | ~2GB |
| Qwen3-VL-8B-Instruct | 8B | High quality | ~12GB | ~7GB | ~4.5GB |
| Qwen3-VL-8B-Thinking | 8B | Advanced CoT | ~12GB | ~7GB | ~4.5GB |
| Qwen3-VL-32B-Instruct | 32B | Best quality | ~28GB | ~14GB | ~8.5GB |
| Qwen3-VL-32B-Thinking | 32B | Complex reasoning | ~28GB | ~14GB | ~8.5GB |
| Qwen2.5-VL-3B-Instruct | 3B | Previous gen | ~6GB | ~3.5GB | ~2GB |
| Qwen2.5-VL-7B-Instruct | 7B | Previous gen | ~15GB | ~8.5GB | ~5GB |

**FP8 Pre-Quantized Models** (40-series GPU recommended):
- `Qwen3-VL-2B-*-FP8`: ~2.5GB VRAM
- `Qwen3-VL-4B-*-FP8`: ~2.5GB VRAM
- `Qwen3-VL-8B-*-FP8`: ~7.5GB VRAM
- `Qwen3-VL-32B-*-FP8`: ~24GB VRAM

### GGUF Quantized Models

| Model | Variants | Features |
|-------|----------|----------|
| Qwen3-VL-4B-Instruct-GGUF | Q4_K_M, Q8_0, F16 | Instruct tuned |
| Qwen3-VL-8B-Instruct-GGUF | Q4_K_M, Q8_0, F16 | Instruct tuned |
| Qwen3-VL-4B-Thinking-GGUF | Q4_K_M, Q8_0, F16 | Thinking mode |
| Qwen3-VL-8B-Thinking-GGUF | Q4_K_M, Q8_0, F16 | Thinking mode |

**GGUF Quantization Guide**:
- `Q4_K_M`: ~3-4GB VRAM, good balance
- `Q8_0`: ~5-7GB VRAM, better quality
- `F16`: ~8-14GB VRAM, best quality

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "QwenVL Utils"
3. Click "Install"
4. Restart ComfyUI

### Method 2: Manual Installation

1. Clone this repository into your `ComfyUI/custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/AkihaTatsu/ComfyUI-QwenVL-Utils.git
   ```

2. Install core dependencies:
   ```bash
   cd ComfyUI-QwenVL-Utils
   pip install -r requirements.txt
   ```

3. **(Optional)** Install optional features:
   ```bash
   # For 4-bit/8-bit quantization
   pip install bitsandbytes>=0.41.0
   
   # For GGUF model support
   pip install llama-cpp-python>=0.2.90
   # Or with CUDA support:
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   
   # For Flash Attention 2 (Ampere+ GPU)
   pip install flash-attn --no-build-isolation
   
   # For SageAttention (experimental)
   pip install sageattention
   
   # Install all optional dependencies
   pip install -e ".[all]"
   ```

4. Restart ComfyUI

### GGUF Setup (Optional)

For GGUF model support with vision capabilities:
- See [ComfyUI-QwenVL GGUF Installation Guide](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md)
- Requires `llama-cpp-python` with `Qwen3VLChatHandler` or `Qwen25VLChatHandler`

## Node Overview

### QwenVL (Basic)
Simplified interface for quick vision-language tasks.

**Parameters**:
- `model_name`: Model selection (HF or GGUF)
- `quantization`: Memory mode (4-bit/8-bit/FP16)
- `attention_mode`: Attention backend (auto/manual)
- `preset_prompt`: Pre-defined task prompts
- `custom_prompt`: Custom text prompt
- `max_tokens`: Maximum output length (64-256000)
- `keep_model_loaded`: Cache model in VRAM
- `seed`: Reproducibility seed

**Inputs**:
- `image` (optional): Single image input
- `video` (optional): Video frames sequence
- `source_path` (optional): File path input

### QwenVL (Advanced)
Full-featured node with granular control.

**Additional Parameters**:
- `temperature`: Sampling randomness (0.0-2.0, default: 0.6)
- `top_p`: Nucleus sampling threshold (0.0-1.0, default: 0.9)
- `num_beams`: Beam search width (1-8, default: 1)
- `repetition_penalty`: Token repetition penalty (0.5-2.0, default: 1.2)
- `frame_count`: Video frame sampling (1-64, default: 16)
- `device`: Device override (auto/cuda/cpu)
- `use_torch_compile`: Enable JIT compilation (default: False)
- **HF-specific**:
  - `min_pixels`: Minimum image resolution (default: 256×28×28 = 200,704)
  - `max_pixels`: Maximum image resolution (default: 1280×28×28 = 1,003,520)
- **GGUF-specific**:
  - `ctx`: Context length (default: 8192, range: 1024-262144)
  - `n_batch`: Batch size (default: 512, range: 64-32768)
  - `gpu_layers`: GPU layers (-1 = all, default: -1)
  - `image_max_tokens`: Max tokens per image (default: 4096)
  - `top_k`: Top-k sampling (default: 0 = disabled)
  - `pool_size`: Memory pool size (default: 4194304)

### Input Utility Nodes

#### Load Image Advanced
Loads images with additional outputs for mask and file path.

**Returns**:
- `image`: Image tensor
- `mask`: Alpha channel mask
- `path`: File path string

**Features**:
- Supports animated images (GIF)
- Auto EXIF orientation
- Multiple image formats (JPG, PNG, BMP, TIFF, WebP, GIF)

#### Load Video Advanced
Loads video files from ComfyUI input directory.

**Returns**:
- `video`: Video object
- `path`: File path string

#### Load Video Advanced (Path)
Loads video files from custom file path string.

**Input**:
- `file`: File path string (e.g., "X://path/to/video.mp4")

**Returns**:
- `video`: Video object
- `path`: File path string

#### Multiple Paths Input
Creates a path batch from multiple image/video files.

**Parameters**:
- `inputcount`: Number of input paths (1-1000)
- `path_1`, `path_2`, ...: Individual file paths
- `sample_fps`: Video sampling FPS (default: 1)
- `max_frames`: Maximum frames per video (default: 2)
- `use_total_frames`: Use all video frames (default: True)
- `use_original_fps_as_sample_fps`: Use original video FPS (default: True)

**Returns**:
- `paths`: List of path objects for batch processing

**Supported Formats**:
- Images: JPG, JPEG, PNG, BMP, TIFF, WebP, GIF
- Videos: MP4, MKV, MOV, AVI, FLV, WMV, WebM, M4V

## Usage Guide

**For High VRAM Systems (16GB+)**:
```
quantization: None (FP16)
attention_mode: flash_attention_2
keep_model_loaded: True
use_torch_compile: True
```

**For Low VRAM Systems (<8GB)**:
```
quantization: 4-bit (VRAM-friendly)
attention_mode: auto
keep_model_loaded: False
Use GGUF models with Q4_K_M quantization
```

**For Video Analysis**:
```
frame_count: 16-32 (balance detail vs. speed)
max_tokens: 2048-4096 (longer outputs)
```

### Preset Prompts

| Prompt | Use Case | Output Type |
|--------|----------|-------------|
| ❌ **None** | No system prompt | Custom only |
| 🖼️ **Tags** | Generate comma-separated tags | Short list |
| 🖼️ **Simple Description** | One-sentence summary | 1 sentence |
| 🖼️ **Detailed Description** | Comprehensive paragraph | 6-10 sentences |
| 🖼️ **Ultra Detailed Description** | Exhaustive analysis | 10-16 sentences |
| 🎬 **Cinematic Description** | Film-style description | Atmospheric |
| 🖼️ **Detailed Analysis** | Structured breakdown | Categorized |
| 📹 **Video Summary** | Video content summary | Narrative |
| 📖 **Short Story** | Creative storytelling | Fiction |
| 🪄 **Prompt Refine & Expand** | Enhance T2I prompts | Enhanced text |

## Attention Mode Selection

### Auto Mode Priority
When `attention_mode: auto`, the system selects in this order:

1. **Flash Attention 2** (`flash_attention_2`): 
   - Best raw performance
   - Requires: Ampere+ GPU (RTX 30xx/40xx, A100, H100)
   - Install: `pip install flash-attn --no-build-isolation`
   
2. **SDPA Flash** (`sdpa_flash`):
   - PyTorch 2.0+ built-in Flash backend
   - Excellent performance with better compatibility (recommended)
   - Supports newer architectures (Blackwell, etc.)
   - Requires: Ampere+ GPU
   
3. **SageAttention** (`sage_attention`):
   - Memory efficient wrapper
   - Experimental feature
   - Install: `pip install sageattention`
   
4. **SDPA Math** (`sdpa_math`):
   - PyTorch SDPA with math backend
   - Slower but more stable fallback
   
5. **Eager** (`eager`):
   - Standard PyTorch attention
   - Always available (slowest)

### Manual Selection
Force specific backend by setting `attention_mode`:
- `auto`: Auto-select best available (recommended)
- `flash_attention_2`: External flash-attn package
- `sdpa_flash`: PyTorch SDPA with Flash backend
- `sdpa_math`: PyTorch SDPA with math backend (disable Flash)
- `sdpa`: Legacy option (auto-selects Flash or math)
- `sage_attention`: SageAttention wrapper
- `eager`: Standard PyTorch attention

## Performance Benchmarks

### Inference Speed (approximate)

| Model | GPU | Quantization | Tokens/sec | VRAM Usage |
|-------|-----|--------------|------------|------------|
| Qwen3-VL-4B | RTX 4090 | FP16 | ~120 | 10GB |
| Qwen3-VL-4B | RTX 4090 | 8-bit | ~100 | 6GB |
| Qwen3-VL-4B | RTX 4090 | 4-bit | ~80 | 4GB |
| Qwen3-VL-8B | RTX 4090 | 8-bit | ~70 | 10GB |
| Qwen3-VL-4B-GGUF | RTX 4090 | Q4_K_M | ~90 | 4GB |

*Benchmarks vary based on image resolution, prompt length, and hardware.*

## Troubleshooting

### Common Issues

**"Out of Memory" Error**:
- Solution 1: Use lower quantization (8-bit → 4-bit)
- Solution 2: Use GGUF models with Q4_K_M
- Solution 3: Disable `keep_model_loaded`
- Solution 4: Close other applications
- Solution 5: Use smaller model (8B → 4B → 2B)

**"ImportError: transformers"**:
```bash
pip install transformers>=4.37.0
```

**"llama-cpp-python not found" (GGUF)**:
```bash
# CUDA support:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
# CPU only:
pip install llama-cpp-python
```

**"Flash Attention not available"**:
- Requires Ampere or newer GPU (compute capability 8.0+)
- Install: `pip install flash-attn --no-build-isolation`
- Fallback: System will use SDPA automatically

**Model Download Fails**:
- Check internet connection
- Set HuggingFace token: `huggingface-cli login`
- Manual download from [Hugging Face](https://huggingface.co/Qwen)
- Place in `ComfyUI/models/LLM/Qwen-VL/<model_name>/`
