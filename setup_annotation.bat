@echo off
setlocal
chcp 65001 >nul 2>&1
title LocalMuse — Annotation Setup

:: NOTE: setup.bat now detects an NVIDIA GPU (via nvidia-smi) and
:: installs CUDA 12.4 torch directly, and also installs
:: qwen-vl-utils/accelerate from requirements-optional.txt. So on a
:: fresh setup.bat run on a GPU machine, this script has nothing left
:: to do (Steps 2 and 4 below become no-ops - pip sees everything is
:: already satisfied).
::
:: This script remains useful as a REPAIR / VERIFY tool for:
::   - venvs created when no GPU/driver was detected by setup.bat
::     (torch is CPU-only) - Step 2 force-reinstalls CUDA torch.
::   - older venvs created before requirements-optional.txt existed
::     - Step 4 re-checks/installs qwen-vl-utils + accelerate.

echo ============================================================
echo   LocalMuse V2 — Annotation Model Setup
echo ============================================================
echo.

:: ── Find Python ─────────────────────────────────────────────────
set VENV_PY=.\venv\Scripts\python.exe
set PIP=.\venv\Scripts\pip.exe
if not exist "%VENV_PY%" (
    echo [ERROR] venv not found. Run setup.bat first.
    pause & exit /b 1
)

:: ── Step 1: Check current torch ──────────────────────────────────
echo [1/4] Checking current torch installation...
"%VENV_PY%" -c "import torch; print('  Current torch:', torch.__version__)"
"%VENV_PY%" -c "import torch; print('  CUDA available:', torch.cuda.is_available())"
echo.

:: ── Step 2: Install / upgrade to CUDA torch ──────────────────────
echo [2/4] Ensuring torch with CUDA 12.4 support (RTX 3090)...
"%VENV_PY%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
    echo   torch CUDA already available - skipping reinstall.
    goto verify_cuda
)
echo   This may take a while (downloading ~2.5 GB)...
echo.
"%PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall --quiet
if errorlevel 1 (
    echo [ERROR] torch CUDA install failed.
    echo   Check your internet connection and try again.
    pause & exit /b 1
)
echo   torch CUDA installed.

:: ── Step 3: Verify CUDA is now available ─────────────────────────
:verify_cuda
echo.
echo [3/4] Verifying CUDA support...
"%VENV_PY%" -c "import torch; print('  torch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
"%VENV_PY%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] CUDA still not detected after install.
    echo   Possible causes:
    echo     - NVIDIA driver too old (need driver ^>= 525 for CUDA 12.x)
    echo     - CUDA runtime not in PATH
    echo   Annotation will continue on CPU (much slower).
    echo.
)

:: ── Step 4: Install VLM annotation dependencies ──────────────────
echo.
echo [4/4] Installing VLM annotation dependencies...

echo   Installing qwen-vl-utils...
"%PIP%" install "qwen-vl-utils>=0.0.8" --quiet
if errorlevel 1 (echo [ERROR] qwen-vl-utils install failed & pause & exit /b 1)
echo   OK

echo   Installing accelerate...
"%PIP%" install "accelerate>=0.26.0" --quiet
if errorlevel 1 (echo [ERROR] accelerate install failed & pause & exit /b 1)
echo   OK

:: ── Summary ──────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Setup complete. Final environment:
"%VENV_PY%" -c "import torch; print('  torch      :', torch.__version__); print('  CUDA       :', torch.cuda.is_available()); print('  GPU        :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
"%VENV_PY%" -c "import qwen_vl_utils; print('  qwen-vl-utils: OK')"
"%VENV_PY%" -c "import accelerate; print('  accelerate :', accelerate.__version__)"
"%VENV_PY%" -c "import transformers; assert hasattr(transformers, 'Qwen3VLForConditionalGeneration'); print('  Qwen3-VL   : OK')"
echo.
echo   Model weights (~15 GB) are downloaded automatically
echo   on first annotation run from Hugging Face Hub.
echo ============================================================
echo.
pause
