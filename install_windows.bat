@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo ================================================================
echo                 AI Mosaic Builder - Installer
echo ================================================================
echo.

where py >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python Launcher ^(py^) was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/windows/
    echo Make sure the Python Launcher and pip are enabled.
    pause
    exit /b 1
)

py -3.12 -c "import sys; print('Using Python', sys.version)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.12 was not found.
    echo.
    echo AI Mosaic Builder recommends Python 3.12 for the pre-built llama.cpp wheels.
    echo Install Python 3.12 and run this installer again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/6] Creating virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto :fail
) else (
    echo [1/6] Existing virtual environment found.
)

set "PY=.venv\Scripts\python.exe"

if exist "%TEMP%\aimosaic_requirements.txt" del /q "%TEMP%\aimosaic_requirements.txt" >nul 2>&1
findstr /V /I /C:"llama-cpp-python" requirements.txt > "%TEMP%\aimosaic_requirements.txt"

:choose_runtime
echo.
echo ================================================================
echo Select llama.cpp runtime:
echo.
echo   1 - CPU only

echo   2 - NVIDIA CUDA 11.8

echo   3 - NVIDIA CUDA 12.1

echo   4 - NVIDIA CUDA 12.4

echo   5 - NVIDIA CUDA 13.0

echo   6 - NVIDIA CUDA 13.2

echo.
set /p CUDA_CHOICE="Choice [1-6, default 1]: "
if "%CUDA_CHOICE%"=="" set "CUDA_CHOICE=1"

set "LLAMA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cpu"
if "%CUDA_CHOICE%"=="2" set "LLAMA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu118"
if "%CUDA_CHOICE%"=="3" set "LLAMA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu121"
if "%CUDA_CHOICE%"=="4" set "LLAMA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu124"
if "%CUDA_CHOICE%"=="5" set "LLAMA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu130"
if "%CUDA_CHOICE%"=="6" set "LLAMA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu132"

if not "%CUDA_CHOICE%"=="1" if not "%CUDA_CHOICE%"=="2" if not "%CUDA_CHOICE%"=="3" if not "%CUDA_CHOICE%"=="4" if not "%CUDA_CHOICE%"=="5" if not "%CUDA_CHOICE%"=="6" (
    echo [ERROR] Invalid choice.
    set "CUDA_CHOICE="
    goto :choose_runtime
)

echo.
echo [2/6] Upgrading pip, setuptools and wheel...
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [3/6] Installing common Python dependencies...
"%PY%" -m pip install -r "%TEMP%\aimosaic_requirements.txt"
if errorlevel 1 goto :fail

echo [4/6] Installing llama-cpp-python from:
echo        %LLAMA_INDEX%
"%PY%" -m pip uninstall -y llama-cpp-python >nul 2>&1
"%PY%" -m pip install "llama-cpp-python^>=0.3.33^,^<0.4" --extra-index-url "%LLAMA_INDEX%"
if errorlevel 1 (
    echo.
    echo [ERROR] Pre-built llama-cpp-python installation failed.
    echo If your selected CUDA wheel is unavailable for your Python/CUDA combination,
    echo use CPU mode or install llama-cpp-python manually with a supported build.
    goto :fail
)

echo [5/6] Verifying installed packages...
"%PY%" -c "import PySide6, PIL, numpy, cv2, imagehash, ultralytics, llama_cpp; print('All Python packages imported successfully.')"
if errorlevel 1 goto :fail

if not "%CUDA_CHOICE%"=="1" "%PY%" -c "from llama_cpp import llama_supports_gpu_offload; print('llama.cpp GPU offload:', llama_supports_gpu_offload())" 2>nul

echo [6/6] Checking application startup imports...
"%PY%" -c "import main; print('Application imports successfully.')"
if errorlevel 1 goto :fail

if exist "%TEMP%\aimosaic_requirements.txt" del /q "%TEMP%\aimosaic_requirements.txt" >nul 2>&1

echo.
echo ================================================================
echo Installation completed successfully.
echo ================================================================
echo.
echo Activate the environment:
echo     .venv\Scripts\activate
 echo.
echo Start AI Mosaic Builder:
echo     .venv\Scripts\python.exe main.py
 echo.
echo Debug mode:
echo     .venv\Scripts\python.exe main.py --debug
 echo.
pause
exit /b 0

:fail
if exist "%TEMP%\aimosaic_requirements.txt" del /q "%TEMP%\aimosaic_requirements.txt" >nul 2>&1
echo.
echo ================================================================
echo Installation FAILED.
echo ================================================================
echo Check the error above, then run this installer again.
echo The existing .venv has been preserved so you can retry.
pause
exit /b 1
