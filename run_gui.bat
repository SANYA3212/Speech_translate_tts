@echo off
setlocal

echo #############################################################
echo # Launching Speech-Translate-TTS GUI...                   #
echo #############################################################

set VENV_DIR=venv311
set SCRIPT_NAME=speech_translate_tts.py
set PYTHON_EXE="%VENV_DIR%\Scripts\python.exe"

REM Check if venv's python executable exists
if not exist %PYTHON_EXE% (
    echo Error: Python executable not found at %PYTHON_EXE%
    echo Please run setup.bat first to create the environment and install dependencies.
    pause
    exit /b 1
)

REM Check if script exists
if not exist "%SCRIPT_NAME%" (
    echo Error: Main script '%SCRIPT_NAME%' not found in the current directory.
    pause
    exit /b 1
)

REM Activate venv and run the script
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo --- Starting the Python GUI script with explicit interpreter path... ---
%PYTHON_EXE% -X utf8 -B %SCRIPT_NAME%

echo.
echo --- Script Finished ---
echo The application window has been closed or an error occurred.
echo The console window will remain open for inspection.
pause
