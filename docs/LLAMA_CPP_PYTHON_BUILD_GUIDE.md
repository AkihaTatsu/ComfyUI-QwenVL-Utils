# Building llama-cpp-python from Source (with Qwen3-VL / CUDA Support)

This guide covers how to compile and install **llama-cpp-python** with full Qwen3-VL vision-language support and CUDA GPU acceleration. It is primarily written for Windows users, but Linux/macOS instructions are also included.

---

## Table of Contents

- [Background](#background)
- [Why Build from Source?](#why-build-from-source)
- [Prerequisites](#prerequisites)
- [Step 1 — Clone the Source Code](#step-1--clone-the-source-code)
- [Step 2 — Prepare the llama.cpp Submodule](#step-2--prepare-the-llamacpp-submodule)
- [Step 3 — Install Python Build Dependencies](#step-3--install-python-build-dependencies)
- [Step 4 — Configure Environment Variables](#step-4--configure-environment-variables)
  - [Windows (CUDA + MSVC)](#windows-cuda--msvc)
  - [Linux (CUDA + GCC)](#linux-cuda--gcc)
  - [macOS (Metal)](#macos-metal)
  - [CPU-Only (Any Platform)](#cpu-only-any-platform)
- [Step 5 — Build and Install](#step-5--build-and-install)
- [Step 6 — Verify Installation](#step-6--verify-installation)
- [Quick-Install Alternatives (Pre-built Wheels)](#quick-install-alternatives-pre-built-wheels)
- [Troubleshooting / Q&A](#troubleshooting--qa)

---

## Background

**Qwen3-VL** is a vision-language model from Alibaba. To run Qwen3-VL **GGUF** models, you need:

1. **llama.cpp** — the C/C++ inference engine that includes the Qwen3VL architecture (added in build `b6887`, October 2025, via [PR #16780](https://github.com/ggerganov/llama.cpp/pull/16780)).
2. **llama-cpp-python** — the Python bindings that expose a `Qwen3VLChatHandler` class for multimodal inference.

The **official** `llama-cpp-python` package (`pip install llama-cpp-python`) may **not** include `Qwen3VLChatHandler` yet. The [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) fork (v0.3.27+) provides full Qwen3-VL support with the correct chat handler and multimodal pipeline.

## Why Build from Source?

| Scenario | Recommendation |
|----------|----------------|
| Official pip wheel already has `Qwen3VLChatHandler` | Use the pre-built wheel (no compilation needed) |
| Official pip wheel does NOT have `Qwen3VLChatHandler` | Build from the JamePeng fork |
| You need a specific CUDA architecture (e.g., SM 100 for Blackwell) | Build from source |
| You are on an uncommon platform or toolchain | Build from source |
| You want Qwen2.5-VL only (not Qwen3-VL) | The official wheel works fine |

---

## Prerequisites

### All Platforms

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| **Python** | 3.8+ (3.11 recommended) | Runtime & build host |
| **pip** | 21.0+ | Package installer |
| **Git** | 2.30+ | Clone source repos |
| **CMake** | 3.21+ | Build system generator |

### Windows (CUDA)

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| **Visual Studio Build Tools** | 2019 / 2022 / 2025+ | C/C++ compiler (`cl.exe`) |
| **CUDA Toolkit** | 11.7+ (12.x recommended) | GPU compute |
| **Ninja** | 1.10+ | Build system (avoids VS CUDA toolset issues) |

> **Note**: If you use Visual Studio 2025 or 2026 (MSVC 19.4x / 19.5x), CUDA may report `unsupported Microsoft Visual Studio version`. This can be bypassed — see [Q&A](#q7-cuda-says-unsupported-microsoft-visual-studio-version).

### Linux (CUDA)

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| **GCC / Clang** | GCC 9+ or Clang 11+ | C/C++ compiler |
| **CUDA Toolkit** | 11.7+ | GPU compute |
| **Ninja** (optional) | 1.10+ | Faster builds |

### macOS (Metal)

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| **Xcode Command Line Tools** | 14+ | C/C++ compiler |
| **CMake** | 3.21+ | Build system |

---

## Step 1 — Clone the Source Code

```bash
# Clone the JamePeng fork (has Qwen3VL support)
git clone https://github.com/JamePeng/llama-cpp-python.git
cd llama-cpp-python
```

If the official upstream has merged Qwen3VL support by the time you read this:
```bash
# Clone the official repo instead
git clone https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
```

## Step 2 — Prepare the llama.cpp Submodule

```bash
git submodule update --init --recursive
```

> **Tip**: If submodule cloning fails due to network issues or long paths on Windows, you can manually download the llama.cpp source:
> 1. Check which commit the submodule points to:
>    ```bash
>    git config -f .gitmodules submodule.vendor/llama.cpp.url
>    cat vendor/llama.cpp  # or check .gitmodules for the commit
>    ```
> 2. Download from GitHub as a zip:
>    ```
>    https://github.com/ggerganov/llama.cpp/archive/<COMMIT_HASH>.zip
>    ```
> 3. Extract into `vendor/llama.cpp/` (so that `vendor/llama.cpp/CMakeLists.txt` exists).

## Step 3 — Install Python Build Dependencies

```bash
pip install "scikit-build-core[pyproject]>=0.9.2"
```

This is the Python build backend used by llama-cpp-python. Without it, you will get:

```
BackendUnavailable: Cannot import 'scikit_build_core.build'
```

## Step 4 — Configure Environment Variables

### Windows (CUDA + MSVC)

Open a **Developer Command Prompt** or manually initialize MSVC:

```batch
:: Initialize MSVC (adjust path for your VS version)
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Set CUDA paths (adjust to your CUDA installation)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
set CUDA_HOME=%CUDA_PATH%
set CUDACXX=%CUDA_PATH%\bin\nvcc.exe

:: Set CMake arguments
:: -G Ninja: Use Ninja generator (recommended, avoids VS CUDA toolset issues)
:: -DGGML_CUDA=ON: Enable CUDA backend
:: -DCMAKE_CUDA_ARCHITECTURES: Target GPU architecture
set CMAKE_ARGS=-G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
set FORCE_CMAKE=1
```

**Common CUDA Architecture Values**:

| GPU Family | Architecture | `CMAKE_CUDA_ARCHITECTURES` |
|------------|-------------|---------------------------|
| GTX 10xx (Pascal) | SM 61 | `61` |
| RTX 20xx (Turing) | SM 75 | `75` |
| RTX 30xx (Ampere) | SM 86 | `86` |
| RTX 40xx (Ada Lovelace) | SM 89 | `89` |
| RTX 50xx (Blackwell) | SM 100 | `100` |
| A100 | SM 80 | `80` |
| H100 | SM 90 | `90` |

> **Important**: If your CUDA toolkit is installed in a path with **spaces**, set `CUDACXX` using the Windows 8.3 short path or forward slashes. For example:
> ```batch
> :: Get short path via CMD
> for %I in ("C:\Path With Spaces\CUDA\bin\nvcc.exe") do echo %~sI
> :: Use the short path
> set CUDACXX=C:\PATHWI~1\CUDA\bin\nvcc.exe
> ```
>
> Or use forward slashes:
> ```batch
> set CUDACXX=C:/Path With Spaces/CUDA/bin/nvcc.exe
> ```

### Linux (CUDA + GCC)

```bash
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=$CUDA_PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
export PATH=$CUDA_PATH/bin:$PATH

# Set CMake arguments
export CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89"
export FORCE_CMAKE=1
```

### macOS (Metal)

```bash
# Metal is auto-detected; just enable it
export CMAKE_ARGS="-DGGML_METAL=ON"
export FORCE_CMAKE=1
```

### CPU-Only (Any Platform)

```bash
# No special environment variables needed for CPU-only
# Optionally disable GPU backends explicitly:
export CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_METAL=OFF"
export FORCE_CMAKE=1
```

## Step 5 — Build and Install

From the root of the cloned repository:

```bash
# Build and install into the current Python environment
pip install . --no-build-isolation --verbose
```

**Flags explained**:

| Flag | Meaning |
|------|---------|
| `.` | Build the current directory (the cloned repo) |
| `--no-build-isolation` | Use already-installed build deps (scikit-build-core) instead of creating a temp venv |
| `--verbose` | Show full CMake/compiler output for debugging |

**Alternative install commands**:

```bash
# Force-reinstall (overwrite existing version)
pip install . --no-build-isolation --verbose --force-reinstall --no-deps

# Install to a specific target directory (e.g., portable Python)
pip install . --no-build-isolation --verbose --target=/path/to/site-packages

# Build a wheel without installing (useful for distribution)
pip wheel . --no-build-isolation --verbose -w dist/
```

> **Compilation time**: Expect **10–30 minutes** depending on hardware, CUDA architectures, and number of CPU cores. CUDA kernel compilations are the slowest part.

## Step 6 — Verify Installation

```python
# Check version
python -c "import llama_cpp; print('Version:', llama_cpp.__version__)"

# Check Qwen3VL handler
python -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler; print('Qwen3VLChatHandler: OK')"

# Check GPU support
python -c "import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"
```

Expected output:
```
Version: 0.3.27
Qwen3VLChatHandler: OK
GPU offload: True
```

---

## Quick-Install Alternatives (Pre-built Wheels)

If you do **not** need to build from source, try these first:

### Official Wheels with CUDA (if Qwen3VL is supported)

```bash
# CUDA 12.1
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# CUDA 12.4
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# CPU only
pip install llama-cpp-python
```

### JamePeng Fork Wheels (if available)

Check [JamePeng/llama-cpp-python Releases](https://github.com/JamePeng/llama-cpp-python/releases) for pre-built wheels.

### Verify After Installing

Always check whether `Qwen3VLChatHandler` exists:

```python
python -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler; print('OK')"
```

If you get `ImportError`, the installed version doesn't support Qwen3-VL and you need to build from [source](#step-1--clone-the-source-code).

---

## Troubleshooting / Q&A

### Q1: `BackendUnavailable: Cannot import 'scikit_build_core.build'`

**Cause**: The Python build backend (`scikit-build-core`) is not installed.

**Fix**:
```bash
pip install "scikit-build-core[pyproject]>=0.9.2"
```

Then re-run the build command.

---

### Q2: `No CUDA toolset found` (Windows)

**Cause**: CMake is using the Visual Studio generator, which requires CUDA's MSBuild integration files. These are typically not present in conda/portable CUDA installs.

**Fix**: Use the **Ninja** generator instead of the default Visual Studio generator:

```batch
set CMAKE_ARGS=-G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
```

Make sure `ninja.exe` is on your `PATH`. It ships with:
- Visual Studio Build Tools (under `Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/`)
- conda (`conda install ninja`)
- Standalone: [ninja-build.org](https://ninja-build.org/)

---

### Q3: `CMAKE_CUDA_COMPILER: c:PERSONAL` or broken path

**Cause**: CMake cannot handle paths with **spaces** properly when passed through `CMAKE_ARGS`. Characters after the first space are dropped.

**Fix 1** — Use the `CUDACXX` environment variable (instead of `-DCMAKE_CUDA_COMPILER` in `CMAKE_ARGS`):
```batch
set CUDACXX=C:/Path/To/nvcc.exe
```

**Fix 2** — Use the Windows 8.3 short path (no spaces):
```batch
:: Find the short path
for %I in ("C:\Path With Spaces\bin\nvcc.exe") do echo %~sI
:: Example output: C:\PATHWI~1\bin\nvcc.exe
set CUDACXX=C:\PATHWI~1\bin\nvcc.exe
```

**Fix 3** — Use forward slashes:
```batch
set CUDACXX=C:/Path With Spaces/bin/nvcc.exe
```

---

### Q4: `unsupported Microsoft Visual Studio version! Only the versions between 2019 and 2022 (inclusive) are supported!`

**Cause**: CUDA Toolkit does not officially support your MSVC version (e.g., VS 2025 / 2026 with MSVC 19.4x / 19.5x).

**Fix**: Add `--allow-unsupported-compiler` to the CUDA flags:

```batch
set CUDAFLAGS=--allow-unsupported-compiler
set CMAKE_ARGS=-G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler
```

> **Warning**: Using an unsupported host compiler _may_ cause compilation failures or incorrect runtime behavior in rare cases. If you experience issues, consider downgrading to Visual Studio 2022 Build Tools.

---

### Q5: `git submodule update` fails (network / long path issues on Windows)

**Cause**: The `llama.cpp` submodule is large, and Windows has path length limits.

**Fix 1** — Enable long paths on Windows:
```batch
git config --system core.longpaths true
```

**Fix 2** — Download the submodule manually as a zip:

1. Check the required commit:
   ```bash
   git ls-tree HEAD vendor/llama.cpp
   ```
2. Download: `https://github.com/ggerganov/llama.cpp/archive/<COMMIT>.zip`
3. Extract into `vendor/llama.cpp/` so that `vendor/llama.cpp/CMakeLists.txt` exists at the root.

---

### Q6: `WARNING: Target directory ... already exists. Specify --upgrade to force replacement.`

**Cause**: Using `pip install --target=...` and the target directory already has an older version installed.

**Fix 1** — Uninstall the old version first:
```bash
pip uninstall llama-cpp-python -y
```

**Fix 2** — Use `--force-reinstall --no-deps`:
```bash
pip install . --no-build-isolation --verbose --force-reinstall --no-deps
```

---

### Q7: Build succeeds but `import llama_cpp` shows wrong version

**Cause**: Python imports from the source directory (`llama_cpp/` in the cloned repo) instead of the installed package.

**Fix**: Run Python from a **different directory** than the cloned repo:
```bash
cd /
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

Or check the actual import path:
```python
python -c "import llama_cpp; print(llama_cpp.__file__)"
```

---

### Q8: Compilation takes very long or hangs

**Cause**: CUDA kernel compilation for multiple architectures is slow.

**Fix**: Only target your specific GPU architecture:
```batch
:: Instead of multiple architectures:
set CMAKE_ARGS=-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
:: Not: -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"
```

Reducing the architecture list can cut build time from 30+ minutes to under 10.

---

### Q9: `Qwen3VLChatHandler` import fails after successful build

**Cause**: You may have built from the official upstream which doesn't include Qwen3VL support yet.

**Fix**: Ensure you cloned from the correct fork:
```bash
git remote -v
# Should show: https://github.com/JamePeng/llama-cpp-python.git
```

Verify the handler exists in the source:
```bash
grep -r "Qwen3VL" llama_cpp/llama_chat_format.py
```

---

### Q10: macOS Metal build fails

**Cause**: Metal SDK or Xcode version issue.

**Fix**:
```bash
# Ensure Xcode CLT is installed
xcode-select --install

# Set Metal explicitly
export CMAKE_ARGS="-DGGML_METAL=ON"
pip install . --no-build-isolation --verbose
```

---

### Q11: `Can't find a Python library` warning from scikit-build-core

**Cause**: This is a benign warning on Windows when building with conda / embedded Python. It does **not** affect the build.

**Action**: Ignore it. The build will proceed normally.

---

### Q12: How to check which CUDA architectures my GPU supports?

Run:
```python
python -c "import torch; print(torch.cuda.get_device_capability())"
# Output example: (8, 9)  → SM 89 → CMAKE_CUDA_ARCHITECTURES=89
```

Or use `nvidia-smi`:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

---

### Q13: How to uninstall and revert to the pre-built wheel?

```bash
# Remove the compiled version
pip uninstall llama-cpp-python -y

# Install the official pre-built wheel
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

---

### Q14: I only need Qwen2.5-VL, not Qwen3-VL. Do I need to build from source?

**No.** The official `llama-cpp-python` package already includes `Qwen25VLChatHandler`. Install normally:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

---

## Reference Links

- [llama-cpp-python (official)](https://github.com/abetlen/llama-cpp-python)
- [JamePeng/llama-cpp-python (Qwen3VL fork)](https://github.com/JamePeng/llama-cpp-python)
- [llama.cpp (C++ engine)](https://github.com/ggerganov/llama.cpp)
- [Qwen3-VL GGUF models on HuggingFace](https://huggingface.co/models?search=qwen3-vl+gguf)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- [Ninja Build System](https://ninja-build.org/)
