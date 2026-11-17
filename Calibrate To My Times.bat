@echo off
setlocal
rem Fit the machining cycle-time calibration to your measured times.
rem Drag your exported feature CSV onto this file, or name it
rem training_table.csv and put it beside this file.
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo.
    echo   The environment has not been created yet.
    echo   Double-click "Run Part Estimator.bat" first.
    echo.
    pause
    exit /b 1
)
set "CSV=%~1"
if "%CSV%"=="" set "CSV=%~dp0training_table.csv"
if not exist "%CSV%" (
    echo.
    echo   Could not find a CSV to read.
    echo     Looked for: %CSV%
    echo.
    echo   Drag your exported CSV onto this file, or put it beside this
    echo   launcher named training_table.csv.
    echo.
    pause
    exit /b 1
)
echo.
"%VPY%" -m partestimator.machining.calibrate "%CSV%"
echo.
pause
exit /b 0
