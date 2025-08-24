@echo off
setlocal

echo #############################################################
echo # Setting up virtual environment for Speech Translator... #
echo #############################################################

set PYTHON_VERSION=3.11
set VENV_DIR=venv311

REM Find Python 3.11
echo Searching for Python %PYTHON_VERSION%...
where python%PYTHON_VERSION%.exe >nul 2>nul
if %errorlevel% == 0 (
    set PYTHON_EXE=python%PYTHON_VERSION%.exe
    goto found_python
)

where python.exe >nul 2>nul
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('python.exe --version') do set "PY_VERSION_OUTPUT=%%i"
    echo Found python.exe, checking version: %PY_VERSION_OUTPUT%
    echo %PY_VERSION_OUTPUT% | find "Python %PYTHON_VERSION%" >nul
    if %errorlevel% == 0 (
        set PYTHON_EXE=python.exe
        goto found_python
    )
)

echo Error: Python %PYTHON_VERSION% not found in PATH.
echo Please install Python %PYTHON_VERSION% and ensure it's added to your PATH.
pause
exit /b 1

:found_python
echo Found Python executable: %PYTHON_EXE%

REM Create virtual environment
if not exist "%VENV_DIR%" (
    echo Creating virtual environment in .\%VENV_DIR%...
    %PYTHON_EXE% -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment .\%VENV_DIR% already exists.
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies from requirements.txt...
call "%VENV_DIR%\Scripts\activate.bat"

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ####################################################################
    echo # ERROR: Failed to install dependencies.                           #
    echo # Please check the error messages above.                           #
    echo # Ensure you have CUDA 12.1 compatible drivers installed.          #
    echo # The console will remain open for inspection.                     #
    echo ####################################################################
    pause
    exit /b 1
)

echo.
echo #############################################################
echo # Setup complete!                                         #
echo # You can now run the application using run_gui.bat       #
echo #############################################################
echo.

deactivate
pause
