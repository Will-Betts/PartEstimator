@echo off
setlocal
rem Runs every automated check against the installed environment and holds
rem the window open so the result can be read. Run once after installing.
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo.
    echo   The environment has not been created yet.
    echo   Double-click "Run Part Estimator.bat" first, then run this.
    echo.
    pause
    exit /b 1
)
"%VPY%" -m partestimator.selftest
echo.
pause
exit /b 0
