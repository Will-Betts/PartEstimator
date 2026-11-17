@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ===========================================================================
rem  Part Estimator -- sheet metal + machining, one window
rem
rem  First run: creates a .venv INSIDE this folder and installs the pinned
rem  dependencies (needs internet ONCE). Every run after that starts the app
rem  straight away, with no console window, and makes no network connection.
rem
rem  Nothing is written outside this folder. No %%TEMP%%, no %%APPDATA%%, no
rem  %%LOCALAPPDATA%%, no registry. Deleting this folder is a complete uninstall.
rem
rem  Supported: CPython 3.11 - 3.13, 64-bit, Windows. 3.12 is the primary
rem  target and is preferred when several are installed.
rem
rem  Path length: keep this folder's path under about 120 characters (e.g.
rem  C:\Tools\PartEstimator). The CAD kernel has deeply nested files and
rem  Windows refuses paths over 260 characters without a registry change.
rem
rem  Usage: double-click, or drop a .step file onto this file to open it.
rem ===========================================================================

set "TOOLDIR=%~dp0"
if "%TOOLDIR:~-1%"=="\" set "TOOLDIR=%TOOLDIR:~0,-1%"
pushd "%TOOLDIR%" || (
    echo Could not enter "%TOOLDIR%".
    goto :fail
)

if "%TOOLDIR:~0,2%"=="\\" (
    echo Running from a network share. A temporary drive letter is in use for
    echo this session only; nothing is installed on the server.
    echo.
)

set "VENV=%TOOLDIR%\.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "VPYW=%VENV%\Scripts\pythonw.exe"
set "LOCK=%TOOLDIR%\installed-lock.txt"
set "REQ=%TOOLDIR%\requirements.txt"
set "STAMP=%VENV%\requirements.sha256"

rem Keep pip's cache inside this folder, and stop pip phoning home to check
rem for a newer pip on every call.
set "PIP_CACHE_DIR=%TOOLDIR%\scratch\pip-cache"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_NO_INPUT=1"
rem Ignore any per-user site-packages so another project's packages can never
rem shadow the pinned ones.
set "PYTHONNOUSERSITE=1"
set "QT_ENABLE_HIGHDPI_SCALING=1"

if not exist "%REQ%" (
    echo   requirements.txt is missing. This launcher must stay in its folder.
    goto :fail
)

rem Fingerprint requirements.txt, so an edit to it triggers a reinstall.
set "REQHASH="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%REQ%" SHA256 2^>nul') do (
    if not defined REQHASH set "REQHASH=%%H"
)
if defined REQHASH set "REQHASH=%REQHASH: =%"
if not defined REQHASH set "REQHASH=nohash"

if not exist "%VPY%" goto :setup
set "HAVE="
if exist "%STAMP%" set /p HAVE=<"%STAMP%"
if not "%HAVE%"=="%REQHASH%" goto :reinstall
goto :haveenv

:reinstall
echo.
echo   requirements.txt has changed since the last install. Updating...
echo.
rem The old lock describes the old requirements; it must not win.
if exist "%LOCK%" del /q "%LOCK%" >nul 2>&1
goto :install

rem -------------------------------------------------------------------------
rem  First run
rem -------------------------------------------------------------------------
:setup
echo.
echo   Part Estimator first-time setup
echo   -------------------------------
echo   Creating a private environment in this folder and installing the pinned
echo   dependencies. This needs an internet connection ONCE and takes several
echo   minutes -- the CAD kernel and the 3D viewer are large downloads.
echo.

call :findpython
if errorlevel 1 goto :nopython

echo   Using: !PYEXE!
"!PYEXE!" -m venv "%VENV%"
if not exist "%VPY%" (
    echo.
    echo   Could not create the virtual environment.
    echo   This usually means the Python install is missing the 'venv' module.
    goto :fail
)

:install
set "REQFILE=%REQ%"
set "FROMLOCK=0"
if exist "%LOCK%" (
    set "REQFILE=%LOCK%"
    set "FROMLOCK=1"
    echo   Installing from installed-lock.txt ^(exact, every package^)...
) else (
    echo   Installing dependencies ^(pinned, wheels only^)...
)
echo.

"%VPY%" -m pip install --only-binary=:all: --require-virtualenv -r "%REQFILE%"
if errorlevel 1 goto :installfailed

if "%FROMLOCK%"=="0" (
    "%VPY%" -m pip freeze > "%LOCK%" 2>nul
    if exist "%LOCK%" echo   Wrote installed-lock.txt -- the exact set that was installed.
)
> "%STAMP%" echo %REQHASH%

if exist "%TOOLDIR%\scratch\pip-cache" rd /s /q "%TOOLDIR%\scratch\pip-cache" 2>nul

echo.
echo   Setup complete. Starting Part Estimator.
echo.

:haveenv
"%VPY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   The environment in this folder exists but does not run. The Python it
    echo   was built from has probably moved or been uninstalled.
    echo.
    echo   Run "Reset Environment.bat", then this file again.
    goto :fail
)

rem pythonw.exe: no console window behind the app. Startup errors are written
rem to last_error.log in this folder and shown in a message box.
if exist "%VPYW%" (
    start "Part Estimator" /D "%TOOLDIR%" "%VPYW%" -m partestimator %*
) else (
    start "Part Estimator" /D "%TOOLDIR%" "%VPY%" -m partestimator %*
)
popd
endlocal & exit /b 0

rem -------------------------------------------------------------------------
rem  Find a usable Python. 3.12 first (primary target), then 3.13, then 3.11.
rem -------------------------------------------------------------------------
:findpython

rem Escape hatches: set PE_PYTHON, or put the full path to python.exe on one
rem line in python-path.txt beside this file.
if defined PE_PYTHON (
    if exist "%PE_PYTHON%" (
        set "PYEXE=%PE_PYTHON%"
        exit /b 0
    )
)
if exist "%TOOLDIR%\python-path.txt" (
    for /f "usebackq delims=" %%P in ("%TOOLDIR%\python-path.txt") do (
        if exist "%%~P" (
            set "PYEXE=%%~P"
            exit /b 0
        )
    )
)

for %%V in (3.12 3.13 3.11) do (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)" 2^>nul') do (
            set "PYEXE=%%P"
            exit /b 0
        )
    )
)

python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info < (3,14) else 1)" >nul 2>&1
if errorlevel 1 goto :findfail
for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if defined PYEXE exit /b 0

:findfail
exit /b 1

:installfailed
echo.
echo   Installation failed. Nothing was installed outside this folder.
echo.
echo   If pip said "no matching distribution", the Python it used has no
echo   prebuilt wheel for one of the pinned packages -- use Python 3.12.
echo   If it mentioned a path being too long, move this folder somewhere with
echo   a shorter path, e.g. C:\Tools\PartEstimator.
echo.
echo   Then run "Reset Environment.bat" and this file again.
goto :fail

:nopython
echo.
echo   No suitable Python was found.
echo.
echo   Part Estimator needs CPython 3.11 - 3.13, 64-bit (3.12 preferred).
echo   Install Python 3.12 from python.org ("install for this user only" needs
echo   no administrator rights), or ask IT for it.
echo.
echo   If Python IS installed but cannot be found, create python-path.txt next
echo   to this file containing one line: the full path to python.exe.
echo.
goto :fail

:fail
popd 2>nul
echo.
pause
endlocal & exit /b 1
