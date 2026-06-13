@echo off
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title LocalMuse - Build Portable Python Runtime

echo ============================================================
echo   LocalMuse V2 - Build Portable Python Runtime
echo ============================================================
echo.
echo   This copies your existing .\venv\ packages plus the base
echo   Python interpreter into a new .\python\ folder.
echo.
echo   Once built, the WHOLE PROJECT FOLDER (minus .\venv\) can be
echo   copied to another PC and run via LocalMuse.bat with no
echo   Python install and no setup step on the new machine.
echo   LocalMuse.bat already prefers .\venv\, then .\python\, so
echo   nothing else needs to change.
echo.
echo   NOTE: Close LocalMuse first if it is currently running, and
echo         make sure you have at least 7 GB of free disk space
echo         (a copy of the venv packages is created, not moved).
echo.

set "ROOT=%~dp0"
set "VENV=%ROOT%venv"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "PYVENV_CFG=%VENV%\pyvenv.cfg"
set "OUT=%ROOT%python"

if not exist "%VENV_PY%" (
    echo [ERROR] .\venv\ not found. Run setup.bat first.
    goto :end
)
if not exist "%PYVENV_CFG%" (
    echo [ERROR] "%PYVENV_CFG%" not found. venv looks broken - run setup.bat again.
    goto :end
)

:: ------------------------------------------------------------
::  Locate the base Python install referenced by venv\pyvenv.cfg
::  (the "home = C:\...\PythonXXX" line)
:: ------------------------------------------------------------
set "BASE_PY="
set "RAWVAL="
for /f "usebackq tokens=1,* delims==" %%A in ("%PYVENV_CFG%") do (
    if /i "%%A"=="home " set "RAWVAL=%%B"
)
if defined RAWVAL set "BASE_PY=%RAWVAL:~1%"

if not defined BASE_PY (
    echo [ERROR] Could not read "home" from "%PYVENV_CFG%".
    goto :end
)
if not exist "%BASE_PY%\python.exe" (
    echo [ERROR] Base Python interpreter not found at:
    echo            %BASE_PY%
    echo          This script must be run on the same machine where
    echo          .\venv\ was created.
    goto :end
)

echo   Base Python interpreter : %BASE_PY%
echo   Packages source         : %VENV%\Lib\site-packages
echo   Output (portable) folder: %OUT%
echo.

if exist "%OUT%" (
    echo [WARN] "%OUT%" already exists.
    choice /M "Delete it and rebuild from scratch"
    if errorlevel 2 goto :end
    echo   Removing previous build...
    rmdir /s /q "%OUT%" 2>nul
)

:: ------------------------------------------------------------
::  Step 1 - copy the base interpreter (standard library, DLLs),
::  excluding its own (mostly empty) site-packages and Scripts.
:: ------------------------------------------------------------
echo.
echo [1/3] Copying Python interpreter and standard library...
robocopy "%BASE_PY%" "%OUT%" /E /XD "site-packages" "Scripts" "__pycache__" /R:2 /W:2 /NFL /NDL /NJH /NJS >nul
if %ERRORLEVEL% GEQ 8 (
    echo [ERROR] Failed to copy base interpreter ^(robocopy exit code %ERRORLEVEL%^).
    goto :end
)
echo        Done.

:: ------------------------------------------------------------
::  Step 2 - copy installed packages from venv (the big step,
::  several GB - this is what takes most of the time).
:: ------------------------------------------------------------
echo.
echo [2/3] Copying installed packages from venv\Lib\site-packages
echo        ^(several GB - this can take a few minutes^)...
robocopy "%VENV%\Lib\site-packages" "%OUT%\Lib\site-packages" /E /MT:8 /R:2 /W:2 /NFL /NDL /NJH /NJS >nul
if %ERRORLEVEL% GEQ 8 (
    echo [ERROR] Failed to copy site-packages ^(robocopy exit code %ERRORLEVEL%^).
    goto :end
)
echo        Done.

:: ------------------------------------------------------------
::  Step 3 - verify the portable interpreter works on its own
:: ------------------------------------------------------------
echo.
echo [3/3] Verifying portable interpreter (no venv involved)...
"%OUT%\python.exe" -c "import sys; print('  Python     :', sys.version.split()[0]); import torch, fastapi, uvicorn, faiss; print('  torch      :', torch.__version__); print('  CUDA avail :', torch.cuda.is_available())"
if errorlevel 1 (
    echo.
    echo [ERROR] Verification failed - the portable interpreter could not
    echo         import the required packages. See the error above.
    goto :end
)

echo.
echo ============================================================
echo   Done. Portable runtime created at:
echo     %OUT%
echo.
echo   Next steps:
echo     - Run LocalMuse.bat as usual to confirm everything still
echo       works (it will keep using .\venv\ as long as that
echo       folder exists).
echo     - To deploy on another PC: copy this whole project folder
echo       there EXCLUDING the "venv" folder. LocalMuse.bat will
echo       automatically fall back to ".\python\" and run with no
echo       further setup.
echo     - Optional: once you've confirmed ".\python\" works, you
echo       can delete ".\venv\" on THIS PC too, to free up space.
echo       LocalMuse.bat will then use ".\python\" here as well.
echo.
echo   See PORTABLE_DEPLOYMENT.md for details and caveats
echo   (NVIDIA driver requirements, model cache, etc.)
echo ============================================================
:end
echo.
pause
