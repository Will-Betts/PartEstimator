@echo off
setlocal
rem Deletes the installed packages (.venv) so the next launch rebuilds them.
rem Your settings, exports and installed-lock.txt are left alone.
cd /d "%~dp0"
if not exist "%~dp0.venv" (
    echo   Nothing to reset -- there is no .venv folder.
    pause
    exit /b 0
)
echo   Deleting "%~dp0.venv" ...
rd /s /q "%~dp0.venv"
if exist "%~dp0.venv" (
    echo   Could not delete it completely. Close Part Estimator and try again.
) else (
    echo   Done. Double-click "Run Part Estimator.bat" to rebuild.
)
pause
exit /b 0
