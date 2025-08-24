@echo off
setlocal

echo #############################################################
echo # Installing 'requests' into the existing venv...         #
echo # NOTE: This is unlikely to solve the underlying issue.   #
echo #############################################################

set VENV_DIR=venv311

REM Check if venv exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Error: Virtual environment not found at .\%VENV_DIR%
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

REM Activate venv and install the package
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo --- Installing requests ---
python -m pip install requests
if %errorlevel% neq 0 (
    echo ####################################################################
    echo # ERROR: Failed to install requests.                               #
    echo # This confirms the installation process itself is broken.         #
    echo # Please check the error messages above.                           #
    echo ####################################################################
    pause
    exit /b 1
)

echo.
echo #############################################################
echo # 'requests' has been successfully installed.               #
echo # Other packages are likely still missing.                  #
echo #############################################################
echo.

deactivate
pause
