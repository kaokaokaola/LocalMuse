@echo off
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title LocalMuse - Clean Up venv (after portable build)

echo ============================================================
echo   LocalMuse V2 - Clean Up .\venv\
echo ============================================================
echo.
echo   This script helps you safely remove the old .\venv\ folder
echo   (~6 GB) AFTER .\python\ (the portable runtime built by
echo   build_portable_python.bat) has been verified to work on
echo   its own.
echo.
echo   It works in two stages, run on two separate occasions:
echo     1st run: renames .\venv\ to .\venv_old_DELETE_ME\
echo              (reversible - nothing is deleted yet)
echo     2nd run: after you confirm LocalMuse.bat still works,
echo              run this script again to permanently delete
echo              .\venv_old_DELETE_ME\
echo.

set "ROOT=%~dp0"
set "VENV=%ROOT%venv"
set "OLDVENV=%ROOT%venv_old_DELETE_ME"
set "EMBED_PY=%ROOT%python\python.exe"

if not exist "%EMBED_PY%" (
    echo [ERROR] .\python\python.exe not found.
    echo         Run build_portable_python.bat first and make sure
    echo         it completes all 3 steps successfully.
    goto :end
)

if exist "%OLDVENV%" goto :ask_delete

if not exist "%VENV%" (
    echo [INFO] .\venv\ does not exist already - nothing to do.
    goto :end
)

echo   .\python\ was found - looks like the portable build is done.
echo.
echo   STAGE 1: Rename .\venv\ to .\venv_old_DELETE_ME\
echo.
echo   Close LocalMuse first if it is currently running
echo   ^(otherwise the rename below may fail^).
echo.
pause

ren "%VENV%" "venv_old_DELETE_ME"
if errorlevel 1 (
    echo.
    echo [ERROR] Could not rename .\venv\.
    echo         Is LocalMuse.bat still running? Close it and try again.
    goto :end
)

echo.
echo ============================================================
echo   Done. .\venv\ has been renamed to .\venv_old_DELETE_ME\.
echo.
echo   NEXT STEPS:
echo     1. Run LocalMuse.bat now. The first line printed should be:
echo          [OK] Using embedded Python (.\python\)
echo        and the app should start and work normally - check
echo        search, semantic/Chinese search, and Annotate if you
echo        use it.
echo.
echo     2. If everything works: run cleanup_venv.bat again to
echo        PERMANENTLY delete .\venv_old_DELETE_ME\ (~6 GB freed).
echo.
echo     3. If something is broken: close LocalMuse, rename
echo        .\venv_old_DELETE_ME\ back to .\venv\ (File Explorer or
echo        "ren venv_old_DELETE_ME venv"), and report the issue.
echo ============================================================
goto :end

:ask_delete
echo   Found .\venv_old_DELETE_ME\ from a previous run of this script.
echo.
echo   Only continue if you have ALREADY confirmed LocalMuse.bat
echo   works correctly using .\python\ (it should print
echo   "Using embedded Python (.\python\)" and run normally).
echo.
echo   This next step PERMANENTLY DELETES .\venv_old_DELETE_ME\
echo   (~6 GB) and cannot be undone.
echo.
choice /M "Permanently delete .\venv_old_DELETE_ME\ now"
if errorlevel 2 (
    echo.
    echo [INFO] Nothing deleted. Re-run this script later when ready,
    echo        or rename .\venv_old_DELETE_ME\ back to .\venv\ to
    echo        restore it.
    goto :end
)

echo.
echo   Deleting .\venv_old_DELETE_ME\  ...  please wait.
rmdir /s /q "%OLDVENV%"
if exist "%OLDVENV%" (
    echo.
    echo [ERROR] Could not fully delete .\venv_old_DELETE_ME\.
    echo         Some files may be in use - close LocalMuse and any
    echo         editors/terminals pointing into that folder, then
    echo         run this script again.
    goto :end
)
echo.
echo ============================================================
echo   Done. .\venv_old_DELETE_ME\ deleted - about 6 GB freed.
echo   LocalMuse.bat will keep using .\python\ from now on.
echo ============================================================

:end
echo.
pause
