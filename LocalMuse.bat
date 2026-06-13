@echo off
chcp 65001 >nul
title LocalMuse V2
cd /d "%~dp0"

set "ROOT=%~dp0"
set "VENV_PY=%ROOT%venv\Scripts\python.exe"
set "EMBED_PY=%ROOT%python\python.exe"
set "MAIN_PY=%ROOT%main.py"
set "CRASH_LOG=%ROOT%localmuse_crash.log"
set PYTHON_EXE=

:: Priority: venv > embedded > system PATH
if exist "%VENV_PY%" (
    set "PYTHON_EXE=%VENV_PY%"
    echo [OK] Using venv Python (.\venv\)
    goto :check
)

if exist "%EMBED_PY%" (
    set "PYTHON_EXE=%EMBED_PY%"
    echo [OK] Using embedded Python (.\python\)
    goto :check
)

python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_EXE=python
    echo [OK] Using system Python
    goto :check
)

echo.
echo [ERROR] Python not found. Please run setup.bat first.
goto :end

:check
"%PYTHON_EXE%" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] FastAPI / uvicorn not installed. Please run setup.bat.
    goto :end
)

"%PYTHON_EXE%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch not installed. CLIP search requires PyTorch.
    echo        Please run setup.bat.
    goto :end
)

"%PYTHON_EXE%" -c "import faiss" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] faiss-cpu not installed. Please run setup.bat.
    goto :end
)

if exist "%CRASH_LOG%" del "%CRASH_LOG%" >nul 2>&1

echo Starting LocalMuse V2...
echo Browser will open at: http://127.0.0.1:17788
echo Press Ctrl+C to quit.
echo.

"%PYTHON_EXE%" "%MAIN_PY%" %* 2>&1

if exist "%CRASH_LOG%" (
    echo.
    echo [ERROR] LocalMuse crashed. Error details:
    echo ============================================================
    type "%CRASH_LOG%"
    echo ============================================================
    echo.
    echo Crash log saved to: %CRASH_LOG%
    echo Try running setup.bat to repair dependencies.
)

:end
echo.
pause
