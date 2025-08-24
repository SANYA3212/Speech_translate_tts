@echo off
setlocal

echo #############################################################
echo # Setting up virtual environment and downloading models...  #
echo #############################################################

set PYTHON_VERSION=3.11
set VENV_DIR=venv311
set PYTHON_LAUNCHER=py -%PYTHON_VERSION%

REM Check for Python 3.11 via the Python Launcher
%PYTHON_LAUNCHER% -c "import sys; sys.exit(0)" >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python %PYTHON_VERSION% not found.
    echo Please install Python %PYTHON_VERSION% for Windows and ensure the 'py' launcher is in your PATH.
    echo You can get it from python.org.
    pause
    exit /b 1
)

echo Found Python %PYTHON_VERSION% via 'py' launcher.

REM Force-delete existing venv directory to ensure a clean slate
if exist "%VENV_DIR%" (
    echo Deleting existing virtual environment directory to ensure a clean setup...
    rd /s /q "%VENV_DIR%"
)

REM Create virtual environment
echo Creating virtual environment in .\%VENV_DIR%...
%PYTHON_LAUNCHER% -m venv %VENV_DIR%
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies from requirements.txt...
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo --- Ensuring pip is run by the correct Python interpreter from venv... ---
echo --- Forcing re-download of all packages to avoid cache issues... ---
python -m pip install --no-cache-dir -r requirements.txt
if %errorlevel% neq 0 (
    echo ####################################################################
    echo # ERROR: Failed to install dependencies.                           #
    echo # Please check the error messages above.                           #
    echo # The console will remain open for inspection.                     #
    echo ####################################################################
    pause
    exit /b 1
)

echo.
echo #############################################################
echo # Dependencies installed. Now downloading models...         #
echo #############################################################

REM Run the model downloader script using the venv's python
python download_models.py
if %errorlevel% neq 0 (
    echo ####################################################################
    echo # ERROR: Failed to download models.                                #
    echo # Please check the error messages above.                           #
    echo # The console will remain open for inspection.                     #
    echo ####################################################################
    pause
    exit /b 1
)

echo.
echo #############################################################
echo # Setup complete! Models and dependencies are ready.        #
echo # You can now run the application using run_gui.bat       #
echo #############################################################
echo.

deactivate
pause
