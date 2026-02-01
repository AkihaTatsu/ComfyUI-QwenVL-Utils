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

## Features

### Dual Backend Support
* **HuggingFace Transformers**: High-quality inference with full model support
* **GGUF (llama-cpp-python)**: Lower VRAM usage with quantized models
* Automatic backend selection based on model type

### Smart Attention Backends
* **Automatic Selection**: Intelligent priority chain based on hardware
  1. Flash Attention 2 (fastest, requires Ampere+ GPU)
  2. SageAttention (memory efficient, experimental)
  3. SDPA (PyTorch 2.0+ built-in)
  4. Eager (fallback, always available)
* **Manual Override**: Force specific attention implementation when needed

### Performance Optimization
* **Memory Management**: Automatic quantization downgrade when VRAM insufficient
* **Model Caching**: Optional "Keep Model Loaded" for faster sequential runs
* **Torch Compile**: Optional JIT compilation for improved inference speed
* **Low CPU Memory Mode**: Efficient loading with minimal system RAM usage

### User Experience
* **Two Node Variants**:
  - `QwenVL (Basic)`: Simplified interface for quick tasks
  - `QwenVL (Advanced)`: Full parameter control for power users
* **Preset Prompts**: 9 built-in prompts for common vision tasks
* **Custom Prompts**: Full flexibility with user-defined prompts
* **Seed Support**: Reproducible generation with deterministic outputs

### Robust Error Handling
* Detailed error messages with root cause analysis
* Automatic dependency checking at startup
* Installation commands for missing packages
* Hardware compatibility warnings
* Memory requirement estimates

### Compatibility
* **Model Path Sharing**: Uses same directories as ComfyUI-QwenVL
  - HuggingFace: `ComfyUI/models/LLM/Qwen-VL/`
  - GGUF: `ComfyUI/models/LLM/GGUF/`
* **Configuration Compatibility**: Reads ComfyUI-QwenVL config files
* **Cross-Platform**: Windows, Linux, macOS support

## Supported Models

### HuggingFace Vision-Language Models

| Model | Size | Features | VRAM (FP16) | VRAM (8-bit) | VRAM (4-bit) |
|-------|------|----------|-------------|--------------|--------------|
| Qwen3-VL-2B-Instruct | 2B | General VL | ~6GB | ~4GB | ~3GB |
| Qwen3-VL-2B-Thinking | 2B | CoT reasoning | ~6GB | ~4GB | ~3GB |
| Qwen3-VL-4B-Instruct | 4B | Balanced | ~10GB | ~6GB | ~4GB |
| Qwen3-VL-4B-Thinking | 4B | CoT reasoning | ~10GB | ~6GB | ~4GB |
| Qwen3-VL-8B-Instruct | 8B | High quality | ~18GB | ~10GB | ~6GB |
| Qwen3-VL-8B-Thinking | 8B | Advanced CoT | ~18GB | ~10GB | ~6GB |
| Qwen3-VL-32B-Instruct | 32B | Best quality | ~70GB | ~36GB | ~20GB |
| Qwen3-VL-32B-Thinking | 32B | Complex reasoning | ~70GB | ~36GB | ~20GB |
| Qwen2.5-VL-3B-Instruct | 3B | Previous gen | ~8GB | ~5GB | ~4GB |
| Qwen2.5-VL-7B-Instruct | 7B | Previous gen | ~16GB | ~9GB | ~6GB |

**FP8 Pre-Quantized Models** (40-series GPU recommended):
- All models available with `-FP8` suffix for ~50% VRAM reduction

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
- `temperature`: Sampling randomness (0.0-2.0)
- `top_p`: Nucleus sampling threshold (0.0-1.0)
- `num_beams`: Beam search width (1-8)
- `repetition_penalty`: Token repetition penalty (0.5-2.0)
- `frame_count`: Video frame sampling (1-64)
- `device`: Device override (auto/cuda/cpu)
- `torch_compile`: Enable JIT compilation
- `min_pixels`: Minimum image resolution (HF only)
- `max_pixels`: Maximum image resolution (HF only)
- **GGUF-specific**: `n_ctx`, `n_batch`, `n_gpu_layers`, etc.

## Usage Guide

**For High VRAM Systems (16GB+)**:
```
quantization: None (FP16)
attention_mode: flash_attention_2
keep_model_loaded: True
torch_compile: True
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
| 🖼️ **Tags** | Generate comma-separated tags | Short list |
| 🖼️ **Simple Description** | One-sentence summary | 1 sentence |
| 🖼️ **Detailed Description** | Comprehensive paragraph | 6-10 sentences |
| 🖼️ **Ultra Detailed** | Exhaustive analysis | 10-16 sentences |
| 🎬 **Cinematic** | Film-style description | Atmospheric |
| 🖼️ **Detailed Analysis** | Structured breakdown | Categorized |
| 📹 **Video Summary** | Video content summary | Narrative |
| 📖 **Short Story** | Creative storytelling | Fiction |
| 🪄 **Prompt Refine** | Enhance T2I prompts | Enhanced text |

## Configuration Files

### hf_models.json
Defines HuggingFace models with metadata:
```json
{
  "Qwen3-VL-4B-Instruct": {
    "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
    "type": "instruct",
    "quantized": false,
    "vram_requirement": {
      "full": 10.0,
      "8bit": 6.0,
      "4bit": 4.0
    }
  }
}
```

### gguf_models.json
Defines GGUF model catalog:
```json
{
  "qwenVL_model": [
    {
      "display_name": "Qwen3-VL-4B-Instruct-GGUF",
      "repo_id": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
      "models": {
        "Q4_K_M": {
          "filename": "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
          "mmproj": "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
        }
      }
    }
  ]
}
```

### system_prompts.json
Customizable prompt templates. Add your own prompts by editing this file.

## Attention Mode Selection

### Auto Mode Priority
When `attention_mode: auto`, the system selects in this order:

1. **Flash Attention 2**: 
   - Fastest performance
   - Requires: Ampere+ GPU (RTX 30xx/40xx, A100, H100)
   - Install: `pip install flash-attn --no-build-isolation`
   
2. **SageAttention**:
   - Memory efficient
   - Experimental feature
   - Install: `pip install sageattention`
   
3. **SDPA** (Scaled Dot-Product Attention):
   - PyTorch 2.0+ built-in
   - Good balance of speed and compatibility
   
4. **Eager**:
   - Standard attention
   - Always available (fallback)

### Manual Selection
Force specific backend by setting `attention_mode`:
- `flash_attention_2`: Force Flash Attention 2
- `sage_attention`: Force SageAttention wrapper
- `sdpa`: Force PyTorch SDPA
- `eager`: Force standard attention

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

## About the Models

This extension utilizes the **Qwen-VL** series of models developed by the Qwen Team at Alibaba Cloud. These are powerful, open-source large vision-language models (LVLMs) designed to understand and process both visual and textual information.

**Key Capabilities**:
- Image understanding and description
- Video frame analysis
- Multi-image reasoning
- Chain-of-thought reasoning (Thinking models)
- Long-context processing
- Multilingual support

**Model Variants**:
- **Instruct**: General-purpose vision-language tasks
- **Thinking**: Enhanced reasoning with chain-of-thought
- **FP8**: Pre-quantized for 40-series GPUs

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

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

The Apache License 2.0 is a permissive license that:
- Allows commercial use
- Allows modification and distribution
- Provides patent grant
- Requires preservation of copyright and license notices
- Requires stating changes made to the code

Development Setup:
```bash
git clone https://github.com/AkihaTatsu/ComfyUI-QwenVL-Utils.git
cd ComfyUI-QwenVL-Utils
pip install -e ".[all]"
```
