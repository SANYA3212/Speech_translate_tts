@echo off
setlocal

echo #############################################################
echo # Launching Speech-Translate-TTS GUI...                   #
echo #############################################################

set VENV_DIR=venv311
set SCRIPT_NAME=speech_translate_tts.py

REM Check if venv exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Error: Virtual environment not found at .\%VENV_DIR%
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

echo Starting the Python GUI script...
echo --- Script Output ---

REM Use -X utf8 to force UTF-8 mode, critical for handling different languages.
REM Use -B to prevent Python from writing .pyc files.
python -X utf8 -B %SCRIPT_NAME%

echo.
echo --- Script Finished ---
echo The application window has been closed or an error occurred.
echo The console window will remain open for inspection.
pause
