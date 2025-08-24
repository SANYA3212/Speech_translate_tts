@echo off
setlocal

echo #############################################################
echo # Installing simpleaudio into the existing venv...        #
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
echo --- Installing simpleaudio==1.0.4 ---
python -m pip install simpleaudio==1.0.4
if %errorlevel% neq 0 (
    echo ####################################################################
    echo # ERROR: Failed to install simpleaudio.                            #
    echo # Please check the error messages above.                           #
    echo # The console will remain open for inspection.                     #
    echo ####################################################################
    pause
    exit /b 1
)

echo.
echo #############################################################
echo # simpleaudio has been successfully installed.              #
echo #############################################################
echo.

deactivate
pause
