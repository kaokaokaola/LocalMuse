@echo off
setlocal
chcp 65001 >nul 2>&1
title LocalMuse - Flash Attention Setup

:: NOTE: this script is pinned to torch 2.6.0+cu124 (see the version
:: check below and WHEEL_URL in step 4). If setup.bat's CUDA index
:: ever moves past cu124 / torch is upgraded, update the version
:: check and WHEEL_URL together - see LOCALMUSE_MODIFICATION_GUIDE.md
:: "Dependency Management".

echo ============================================================
echo   LocalMuse V2 - Flash Attention Install (Windows)
echo   Target: flash-attn 2.7.4 / torch 2.6.0+cu124 / Python 3.10
echo ============================================================
echo.

set "VENV_PY=.\venv\Scripts\python.exe"
set "PIP=.\venv\Scripts\pip.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] venv not found. Run setup.bat first.
    pause
    exit /b 1
)

echo [1/4] Checking current environment...
"%VENV_PY%" -c "import sys, torch; print('  python:', sys.version.split()[0]); print('  torch :', torch.__version__); print('  cuda  :', torch.version.cuda); print('  gpu   :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if errorlevel 1 (
    echo [ERROR] Failed to import torch from the venv.
    pause
    exit /b 1
)
echo.

echo [2/4] Verifying compatible versions...
"%VENV_PY%" -c "import sys, torch; ok=(sys.version_info[:2]==(3,10) and torch.__version__.startswith('2.6.0+cu124') and torch.cuda.is_available()); print('  compatible:', ok); raise SystemExit(0 if ok else 1)"
if errorlevel 1 (
    echo [ERROR] This installer expects:
    echo   Python 3.10
    echo   torch 2.6.0+cu124
    echo   CUDA available in torch
    echo.
    echo Current environment is different. Re-run setup_annotation.bat first,
    echo or update WHEEL_URL below to a wheel matching your exact versions.
    pause
    exit /b 1
)
echo.

echo [3/4] Checking whether flash-attn is already installed...
"%VENV_PY%" -c "import flash_attn; print('  flash-attn:', flash_attn.__version__)" >nul 2>&1
if not errorlevel 1 (
    "%VENV_PY%" -c "import flash_attn; print('  flash-attn already installed:', flash_attn.__version__)"
    goto verify
)
echo   flash-attn not installed yet.
echo.

echo [4/4] Installing matching pre-built Windows wheel...
set "WHEEL_URL=https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4+cu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl"
echo   Source: %WHEEL_URL%
echo   Downloading about 187 MB...
"%PIP%" install "%WHEEL_URL%" --no-deps
if errorlevel 1 (
    echo.
    echo [ERROR] flash-attn wheel install failed.
    echo   LocalMuse will still work with PyTorch SDPA fallback.
    echo   Do not run source compilation on Windows unless MSVC + CUDA Toolkit
    echo   are configured; it is slow and often fails.
    pause
    exit /b 1
)

:verify
echo.
echo ============================================================
echo   Verifying flash-attn:
"%VENV_PY%" -c "import flash_attn; print('  flash-attn:', flash_attn.__version__)"
if errorlevel 1 (
    echo [ERROR] flash-attn import failed after install.
    pause
    exit /b 1
)
"%VENV_PY%" -c "import torch; print('  torch:', torch.__version__); print('  CUDA:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo.
echo   Done. flash-attn is installed, but LocalMuse keeps SDPA as the safe
echo   default for Qwen VL annotation on Windows.
echo   To test flash_attention_2 explicitly, set:
echo     set LOCALMUSE_ENABLE_FLASH_ATTN=1
echo ============================================================
echo.
pause
