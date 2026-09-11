# AI Mosaic Builder — Windows Installation

This guide installs AI Mosaic Builder in an isolated `.venv` and selects the correct `llama-cpp-python` CPU/CUDA runtime for the machine.

## Recommended setup

- Windows 10/11
- Python **3.12**
- NVIDIA GPU is optional
- For NVIDIA acceleration, use a supported CUDA llama.cpp wheel
- A Gemma-4 E4B multimodal GGUF model plus matching `mmproj` file

Python 3.12 is the safest choice for the pre-built llama.cpp runtime used by this project. PyTorch's current Windows documentation supports modern Python releases, while llama.cpp wheel availability can be narrower depending on the published build. urlPyTorch Windows installation guidehttps://pytorch.org/get-started/locally/

## One-click installation

From the repository folder, double-click:

```text
install_windows.bat
```

Or from `cmd.exe`:

```bat
install_windows.bat
```

The installer will:

1. Verify Python 3.12.
2. Create `.venv` if it does not exist.
3. Upgrade `pip`, `setuptools`, and `wheel`.
4. Temporarily remove `llama-cpp-python` from the dependency install so pip cannot choose an unsuitable build.
5. Install the remaining dependencies from `requirements.txt`.
6. Ask whether llama.cpp should use CPU or NVIDIA CUDA.
7. Install the selected `llama-cpp-python` wheel.
8. Verify imports for PySide6, Pillow, NumPy, OpenCV, Ultralytics, imagehash, and llama.cpp.
9. Verify that `main.py` imports successfully.

## CUDA choices

The installer provides these llama.cpp wheel choices:

```text
CPU
CUDA 11.8
CUDA 12.1
CUDA 12.4
CUDA 13.0
CUDA 13.2
```

Use a CUDA option only when the NVIDIA driver/runtime on the machine is compatible with that wheel. The upstream llama-cpp-python project publishes pre-built CUDA wheels and also documents source builds with `GGML_CUDA=on`. urlllama-cpp-python installation documentationhttps://github.com/abetlen/llama-cpp-python#installation

If the selected pre-built wheel is unavailable for a particular Python/CUDA combination, use CPU mode or install llama-cpp-python manually using the upstream build instructions.

## Manual installation

Create the environment:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Upgrade packaging tools:

```bat
python -m pip install --upgrade pip setuptools wheel
```

Install the normal dependencies **without llama.cpp**:

```bat
findstr /V /I /C:"llama-cpp-python" requirements.txt > requirements_base.txt
python -m pip install -r requirements_base.txt
```

Then install one llama.cpp runtime.

### CPU llama.cpp

```bat
python -m pip install "llama-cpp-python>=0.3.33,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### NVIDIA CUDA

Replace `cuXXX` with the CUDA wheel you need:

```bat
python -m pip install "llama-cpp-python>=0.3.33,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cuXXX
```

For example:

```bat
python -m pip install "llama-cpp-python>=0.3.33,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

The upstream project also supports building from source with:

```bat
set CMAKE_ARGS=-DGGML_CUDA=on
python -m pip install llama-cpp-python
```

A source build requires an appropriate C/C++ build toolchain and CUDA development environment.

## Why the installer handles llama.cpp separately

`requirements.txt` intentionally records `llama-cpp-python` because it is a real application dependency. However, a generic `pip install -r requirements.txt` does not know whether the machine needs a CPU or CUDA build.

`install_windows.bat` therefore filters that single package temporarily, installs the common dependencies, and then installs the selected CPU/CUDA build explicitly.

```text
requirements.txt
    ↓
Common Python dependencies
    ↓
install_windows.bat selects runtime
    ↓
CPU/CUDA llama.cpp
    ↓
AI Mosaic Builder
```

## Start the application

```bat
.venv\Scripts\python.exe main.py
```

Debug mode:

```bat
.venv\Scripts\python.exe main.py --debug
```

## Model files

The application does not download the multimodal model automatically.

Configure these two files in **Project → Vision Model**:

```text
Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf
```

The model and projector must be a compatible pair.

## Reinstalling

The installer reuses the existing `.venv`. If you need a completely clean environment, delete the `.venv` directory and run `install_windows.bat` again.

Do not delete your application data or source-folder `.aimosaic` directories unless you intentionally want to remove persisted settings/cache/session data.
