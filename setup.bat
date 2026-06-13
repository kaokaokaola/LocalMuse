@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: LocalMuse V2  -  Unified Dependency Installer
:: [Chinese comment removed for cmd.exe compatibility]
::
:: Installs all dependencies into .\venv\ (Python virtual environment).
:: Run this once before launching LocalMuse for the first time.
:: Re-run at any time to repair a broken installation.
::
:: Package versions live in requirements.txt (required) and
:: requirements-optional.txt (optional / graceful-degradation
:: features). To change a dependency version, edit those files -
:: not this script. The only exceptions are torch/torchvision and
:: OpenAI CLIP, which this script installs separately (see Steps
:: 3 and 6 below, and requirements.txt for why).

set "ROOT=%~dp0"
set "VENV=%ROOT%venv"
set "VENV_PY=%ROOT%venv\Scripts\python.exe"
set "VENV_PIP=%ROOT%venv\Scripts\pip.exe"
set "REQ=%ROOT%requirements.txt"
set "REQ_OPT=%ROOT%requirements-optional.txt"

echo.
echo ============================================================
echo   LocalMuse V2  ^|  Dependency Setup  ^(6 steps^)
echo ============================================================
echo.

:: ============================================================
::  Quick check  -  skip install if venv is already fully set up
:: ============================================================
if exist "%VENV_PY%" (
    echo [Check] Verifying existing virtual environment...
    "%VENV_PY%" -c "import fastapi, uvicorn, torch, faiss, cv2" >nul 2>&1
    if not errorlevel 1 (
        echo [OK]   Environment is complete. Skipping installation.
        goto :verify
    )
    echo [INFO]  Some packages are missing. Running full install.
    echo.
)

:: ============================================================
::  Locate Python 3.10+  (py launcher -> PATH)
:: ============================================================
set PYTHON=
for %%V in (3.12 3.11 3.10) do (
    if not defined PYTHON (
        py -%%V --version >nul 2>&1
        if not errorlevel 1 set PYTHON=py -%%V
    )
)
if not defined PYTHON (
    python --version >nul 2>&1
    if not errorlevel 1 set PYTHON=python
)
if not defined PYTHON (
    echo.
    echo [ERROR] Python 3.10 or higher was not found on this system.
    echo         Install Python from: https://www.python.org/downloads/
    echo         Tick "Add Python to PATH" during installation.
    echo.
    goto :done
)
for /f "tokens=*" %%i in ('%PYTHON% --version 2^>^&1') do echo [OK]   Found: %%i

:: ============================================================
::  Check Git  (required for OpenAI CLIP installation)
:: ============================================================
set GIT_OK=0
git --version >nul 2>&1
if not errorlevel 1 (
    set GIT_OK=1
    for /f "tokens=*" %%i in ('git --version 2^>^&1') do echo [OK]   Found: %%i
) else (
    echo [WARN]  Git not found  -  CLIP ^(semantic search^) install will be skipped.
    echo         Install Git from: https://git-scm.com/download/win
)
echo.

:: ============================================================
::  STEP 1/6  Create virtual environment in .\venv\
:: ============================================================
echo [1/6] Creating virtual environment...
if exist "%VENV%" (
    echo        Existing venv found  -  reusing it.
) else (
    %PYTHON% -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        echo         Make sure "python -m venv" is available for your Python install.
        goto :done
    )
    echo [OK]   Created: %VENV%
)
echo.

:: ============================================================
::  STEP 2/6  Upgrade pip, setuptools, wheel
:: ============================================================
echo [2/6] Upgrading pip / setuptools / wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel -q
if errorlevel 1 (
    echo [WARN]  pip upgrade had issues  -  continuing anyway.
) else (
    echo [OK]   pip upgraded.
)
echo.

:: ============================================================
::  STEP 3/6  PyTorch  -  GPU-aware install
::
::  If an NVIDIA GPU is detected (nvidia-smi runs successfully),
::  install the CUDA 12.4 build directly (~2.5 GB). Otherwise
::  install the CPU-only build (~300 MB).
::
::  This installs the right build the first time, so
::  setup_annotation.bat no longer needs to download torch again
::  and force-reinstall it with --force-reinstall in the common
::  case. setup_annotation.bat remains as a repair tool for the
::  case where no GPU/driver was available when setup.bat ran.
:: ============================================================
echo [3/6] Installing PyTorch...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo        NVIDIA GPU detected - installing PyTorch ^(CUDA 12.4, ~2.5 GB^)...
    echo        This may take a while, please wait...
    "%VENV_PY%" -m pip install torch torchvision ^
        --index-url https://download.pytorch.org/whl/cu124 -q
) else (
    echo        No NVIDIA GPU detected - installing PyTorch ^(CPU, ~300 MB^)...
    "%VENV_PY%" -m pip install torch torchvision ^
        --index-url https://download.pytorch.org/whl/cpu -q
)
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch installation failed.
    echo         Check your internet connection and try again.
    goto :done
)
"%VENV_PY%" -c "import torch; print('  torch      :', torch.__version__); print('  CUDA avail :', torch.cuda.is_available())"
echo [OK]   PyTorch installed.
echo.

:: ============================================================
::  STEP 4/6  Required dependencies  (requirements.txt)
::  Web server, image processing, FAISS, etc. Must all succeed
::  for LocalMuse to start.
:: ============================================================
echo [4/6] Installing required dependencies from requirements.txt...
"%VENV_PY%" -m pip install -r "%REQ%" -q
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install one or more REQUIRED packages.
    echo         See requirements.txt. Check your internet connection
    echo         and try again, then re-run setup.bat.
    goto :done
)
echo [OK]   Required dependencies installed.
echo.

:: ============================================================
::  STEP 5/6  Optional dependencies  (requirements-optional.txt)
::  Pose search, OCR, multilingual search, VLM annotation deps.
::  LocalMuse runs without these - the matching feature is simply
::  disabled if a package is missing. Failures here are warnings,
::  not fatal.
:: ============================================================
echo [5/6] Installing optional dependencies from requirements-optional.txt...
"%VENV_PY%" -m pip install -r "%REQ_OPT%" -q
if errorlevel 1 (
    echo [WARN]  One or more optional packages failed to install.
    echo         The matching feature^(s^) ^(pose search / OCR /
    echo         multilingual search / annotation^) may be disabled.
    echo         To retry:
    echo           "%VENV_PY%" -m pip install -r requirements-optional.txt
) else (
    echo [OK]   Optional dependencies installed.
)
echo.

:: ============================================================
::  STEP 6/6  OpenAI CLIP  (requires Git  -  installed from GitHub)
::  Essential for semantic text-to-image search.
::  Not published on PyPI; must be built from source.
:: ============================================================
echo [6/6] Installing OpenAI CLIP ^(semantic search engine^)...
if "%GIT_OK%"=="0" (
    echo [SKIP]  Git not available  -  CLIP skipped.
    echo.
    echo         To enable semantic search later:
    echo           1. Install Git from https://git-scm.com/download/win
    echo           2. "%VENV_PY%" -m pip install git+https://github.com/openai/CLIP.git
    echo.
    goto :verify
)

"%VENV_PY%" -m pip install "git+https://github.com/openai/CLIP.git" -q
if errorlevel 1 (
    echo [WARN]  CLIP install failed  -  semantic text search will be disabled.
    echo.
    echo         To retry:  "%VENV_PY%" -m pip install git+https://github.com/openai/CLIP.git
    echo.
) else (
    echo [OK]   OpenAI CLIP installed.
)
echo.

:: ============================================================
::  VERIFICATION  -  check every installed package
:: ============================================================
:verify
echo ============================================================
echo   Package Verification
echo ============================================================
echo.
set READY=1

:: --- Required (must all pass) ---
"%VENV_PY%" -c "import fastapi;     print('  [OK]  FastAPI       ' + fastapi.__version__)" 2>nul
if errorlevel 1 ( echo   [FAIL] FastAPI       NOT installed & set READY=0 )

"%VENV_PY%" -c "import uvicorn;     print('  [OK]  uvicorn       ' + uvicorn.__version__)" 2>nul
if errorlevel 1 ( echo   [FAIL] uvicorn       NOT installed & set READY=0 )

"%VENV_PY%" -c "import torch;       print('  [OK]  PyTorch       ' + torch.__version__)" 2>nul
if errorlevel 1 ( echo   [FAIL] PyTorch       NOT installed & set READY=0 )

"%VENV_PY%" -c "import torchvision; print('  [OK]  torchvision   ' + torchvision.__version__)" 2>nul
if errorlevel 1 ( echo   [FAIL] torchvision   NOT installed & set READY=0 )

"%VENV_PY%" -c "import numpy;       print('  [OK]  numpy         ' + numpy.__version__)" 2>nul
if errorlevel 1 ( echo   [FAIL] numpy         NOT installed & set READY=0 )

"%VENV_PY%" -c "import cv2;         print('  [OK]  OpenCV        ' + cv2.__version__)" 2>nul
if errorlevel 1 ( echo   [FAIL] OpenCV        NOT installed & set READY=0 )

"%VENV_PY%" -c "import PIL;         print('  [OK]  Pillow        ' + PIL.__version__)" 2>nul
if errorlevel 1 ( echo   [FAIL] Pillow        NOT installed & set READY=0 )

"%VENV_PY%" -c "import faiss;       print('  [OK]  faiss-cpu     OK')" 2>nul
if errorlevel 1 ( echo   [FAIL] faiss-cpu     NOT installed & set READY=0 )

"%VENV_PY%" -c "import ftfy, regex, tqdm, timm; print('  [OK]  ftfy/regex/tqdm/timm  OK')" 2>nul
if errorlevel 1 ( echo   [WARN] ftfy/regex/tqdm/timm  partially missing )

echo.

:: --- Core AI: CLIP ---
"%VENV_PY%" -c "import clip; print('  [OK]  CLIP          OK  ^<-- semantic search ENABLED^>')" 2>nul
if errorlevel 1 ( echo   [WARN] CLIP          NOT installed  ^<-- semantic search DISABLED^> )

:: --- Multilingual: M-CLIP ---
"%VENV_PY%" -c "import multilingual_clip, transformers; print('  [OK]  M-CLIP        OK  ^<-- multilingual search ENABLED^>')" 2>nul
if errorlevel 1 ( echo   [INFO] M-CLIP         not installed  ^<-- English-only search^> )

echo.

:: --- Optional ---
"%VENV_PY%" -c "import ultralytics; print('  [OK]  ultralytics   OK  ^<-- pose search ENABLED^>')" 2>nul
if errorlevel 1 ( echo   [INFO] ultralytics    not installed  ^<-- pose search DISABLED^> )

"%VENV_PY%" -c "import easyocr;     print('  [OK]  easyocr       OK  ^<-- OCR search ENABLED^>')" 2>nul
if errorlevel 1 ( echo   [INFO] easyocr         not installed  ^<-- OCR search DISABLED^> )

"%VENV_PY%" -c "import qwen_vl_utils, accelerate; print('  [OK]  qwen-vl-utils/accelerate  OK  ^<-- VLM annotation ENABLED^>')" 2>nul
if errorlevel 1 ( echo   [INFO] qwen-vl-utils/accelerate  not installed  ^<-- VLM annotation DISABLED^> )

echo.
echo ============================================================
if "%READY%"=="1" (
    echo   [DONE] Setup complete!
    echo          Launch the app:  LocalMuse.bat  or  LocalMuse_EN.bat
) else (
    echo   [!!]  One or more REQUIRED packages failed.
    echo         Review the errors above, then run setup.bat again.
)
echo ============================================================

:done
echo.
pause
