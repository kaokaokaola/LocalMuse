@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: LocalMuse V2  -  Unified Dependency Installer
:: [Chinese comment removed for cmd.exe compatibility]
::
:: Installs all dependencies into .\venv\ (Python virtual environment).
:: Run this once before launching LocalMuse for the first time.
:: Re-run at any time to repair a broken installation.

set "ROOT=%~dp0"
set "VENV=%ROOT%venv"
set "VENV_PY=%ROOT%venv\Scripts\python.exe"
set "VENV_PIP=%ROOT%venv\Scripts\pip.exe"

echo.
echo ============================================================
echo   LocalMuse V2  ^|  Dependency Setup  ^(8 steps^)
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
::  STEP 1/8  Create virtual environment in .\venv\
:: ============================================================
echo [1/8] Creating virtual environment...
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
::  STEP 2/8  Upgrade pip, setuptools, wheel
:: ============================================================
echo [2/8] Upgrading pip / setuptools / wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel -q
if errorlevel 1 (
    echo [WARN]  pip upgrade had issues  -  continuing anyway.
) else (
    echo [OK]   pip upgraded.
)
echo.

:: ============================================================
::  STEP 3/8  PyTorch CPU build  (~300 MB)
::  Pinned to the official CPU wheel index.
::  LocalMuse runs on CPU  -  no CUDA required.
:: ============================================================
echo [3/8] Installing PyTorch ^(CPU^)  -  ~300 MB, please wait...
"%VENV_PY%" -m pip install torch torchvision ^
    --index-url https://download.pytorch.org/whl/cpu -q
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch installation failed.
    echo         Check your internet connection and try again.
    goto :done
)
echo [OK]   PyTorch ^(CPU^) installed.
echo.

:: ============================================================
::  STEP 4/8  Web server  -  FastAPI + uvicorn
:: ============================================================
echo [4/8] Installing web server: FastAPI + uvicorn...
"%VENV_PY%" -m pip install "fastapi>=0.110.0" "uvicorn[standard]>=0.29.0" -q
if errorlevel 1 (
    echo [ERROR] FastAPI / uvicorn installation failed.
    goto :done
)
echo [OK]   FastAPI + uvicorn installed.
echo.

:: ============================================================
::  STEP 5/8  Core ML + image libraries
::  numpy MUST be <2.0  -  CLIP requires the numpy 1.x API.
:: ============================================================
echo [5/8] Installing core ML and image libraries...

echo        Pillow / OpenCV ^(headless^) / numpy ^(pinned ^<2.0^)
"%VENV_PY%" -m pip install ^
    "Pillow>=10.0.0" "opencv-python-headless>=4.8.0" "numpy>=1.24.0,<2.0" -q
if errorlevel 1 (
    echo [ERROR] Image library install failed.
    goto :done
)

echo        faiss-cpu  ^(vector similarity search^)
"%VENV_PY%" -m pip install "faiss-cpu>=1.7.4" -q
if errorlevel 1 (
    echo [ERROR] faiss-cpu install failed.
    goto :done
)

echo        NLP utilities: ftfy regex tqdm timm
"%VENV_PY%" -m pip install "ftfy>=6.1.1" "regex>=2023.0.0" "tqdm>=4.65.0" "timm>=0.9.0" -q
if errorlevel 1 (
    echo [WARN]  Some NLP utilities failed  -  search may be partially affected.
)

echo [OK]   Core ML + image libraries installed.
echo.

:: ============================================================
::  STEP 6/8  Optional: pose search + OCR
::  These packages are NOT required to start LocalMuse.
::  Pose and OCR modalities are simply disabled if absent.
:: ============================================================
echo [6/8] Installing optional features ^(pose search + OCR^)...
set OPTIONAL_FAIL=0

"%VENV_PY%" -m pip install "ultralytics>=8.0.0" -q
if errorlevel 1 (
    echo [WARN]   ultralytics failed  -  YOLOv8 pose search will be disabled.
    set OPTIONAL_FAIL=1
) else (
    echo [OK]     ultralytics ^(YOLOv8 pose search^)
)

"%VENV_PY%" -m pip install "easyocr>=1.7.0" -q
if errorlevel 1 (
    echo [WARN]   easyocr failed  -  OCR text search will be disabled.
    set OPTIONAL_FAIL=1
) else (
    echo [OK]     easyocr ^(in-image text search^)
)

if "%OPTIONAL_FAIL%"=="1" (
    echo.
    echo [INFO]  To retry optional packages later:
    echo        "%VENV_PY%" -m pip install ultralytics easyocr
)
echo.

:: ============================================================
::  STEP 7/8  OpenAI CLIP  (requires Git  -  installed from GitHub)
::  Essential for semantic text-to-image search.
::  Not published on PyPI; must be built from source.
:: ============================================================
echo [7/8] Installing OpenAI CLIP ^(semantic search engine^)...
if "%GIT_OK%"=="0" (
    echo [SKIP]  Git not available  -  CLIP skipped.
    echo.
    echo         To enable semantic search later:
    echo           1. Install Git from https://git-scm.com/download/win
    echo           2. "%VENV_PY%" -m pip install git+https://github.com/openai/CLIP.git
    echo.
    goto :mclip
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
::  STEP 8/8  Optional: Multilingual CLIP (M-CLIP)
::  Enables Chinese, Japanese, Korean and 50+ other languages.
::  Uses the SAME indexed image vectors  -  no re-indexing needed.
:: ============================================================
:mclip
echo [8/8] Installing multilingual search support ^(M-CLIP^)...
echo        Enables Chinese / Japanese / Korean and 50+ language queries.
echo.

"%VENV_PY%" -m pip install "transformers>=4.30.0" "multilingual-clip>=1.0.0" -q
if errorlevel 1 (
    echo [WARN]  M-CLIP install failed  -  search will be English-only.
    echo.
    echo         To retry:  "%VENV_PY%" -m pip install transformers multilingual-clip
) else (
    echo [OK]   M-CLIP installed  -  multilingual search ENABLED.
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
