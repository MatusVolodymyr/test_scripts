@echo off
REM AI Detector Prompt Testing - Easy Setup Script for Windows
REM This script automatically sets up everything needed to run the AI detector testing tool

setlocal enabledelayedexpansion

echo.
echo ==================================================
echo   AI Detector Prompt Testing - Easy Setup
echo ==================================================
echo.

REM Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.7 or later.
    echo [ERROR] Visit: https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo [SUCCESS] Found Python !PYTHON_VERSION!
)

REM Set up virtual environment
echo [INFO] Setting up virtual environment...
if exist "venv" (
    echo [WARNING] Virtual environment already exists. Using existing one...
) else (
    echo [INFO] Creating new virtual environment...
    python -m venv venv
    echo [SUCCESS] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
echo [SUCCESS] Virtual environment activated

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [SUCCESS] Pip upgraded

REM Install dependencies
echo [INFO] Installing Python dependencies...
echo [INFO] This may take a few minutes...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install some dependencies
    echo [ERROR] Try running manually: pip install -r requirements.txt
    pause
    exit /b 1
) else (
    echo [SUCCESS] All dependencies installed successfully
)

REM Set up .env file
echo [INFO] Setting up environment configuration...
if exist ".env" (
    echo [WARNING] .env file already exists. Checking configuration...
    findstr /C:"OPENAI_API_KEY=" .env >nul && findstr /C:"ANTHROPIC_API_KEY=" .env >nul
    if errorlevel 1 (
        echo [WARNING] Some API keys may be missing. Please review your .env file.
    ) else (
        echo [SUCCESS] Environment file looks good
        goto :skip_env_setup
    )
) else (
    echo [INFO] Creating .env file from template...
    copy .env.example .env >nul
    echo [SUCCESS] .env file created
)

echo.
echo ==================================================
echo   🔑 API KEY SETUP REQUIRED
echo ==================================================
echo.
echo You need to add your API keys to the .env file:
echo.
echo 1. OpenAI API Key (for GPT models):
echo    - Get it from: https://platform.openai.com/api-keys
echo    - Add to .env: OPENAI_API_KEY=your-key-here
echo.
echo 2. Anthropic API Key (for Claude models):
echo    - Get it from: https://console.anthropic.com/
echo    - Add to .env: ANTHROPIC_API_KEY=your-key-here
echo.
echo 3. Your AI Detector API URL (if different from localhost):
echo    - Add to .env: DETECTOR_API_URL=http://your-detector-url:8000
echo.
echo The .env file will now open in Notepad. Please add your API keys.
echo.
pause
notepad .env

:skip_env_setup

REM Create a simple run script for Windows
echo [INFO] Creating easy run script...
(
echo @echo off
echo REM Simple test runner - activates venv and runs the test
echo.
echo echo 🚀 Starting AI Detector Test...
echo echo.
echo.
echo REM Activate virtual environment
echo call venv\Scripts\activate.bat
echo.
echo REM Run the test
echo python test_detector_prompts.py --input sample_prompts.csv --generate-and-test
echo.
echo echo.
echo echo ✅ Test completed! Check the test_results/ directory for reports.
echo pause
) > run_test.bat

echo [SUCCESS] Created run_test.bat for easy testing

REM Show usage
echo.
echo ==================================================
echo   🚀 READY TO USE!
echo ==================================================
echo.
echo Your testing environment is set up. Here are some ways to run tests:
echo.
echo 1. Double-click run_test.bat (easiest way)
echo.
echo 2. Or use Command Prompt:
echo    run_test.bat
echo.
echo 3. Manual commands:
echo    venv\Scripts\activate.bat
echo    python test_detector_prompts.py --input sample_prompts.csv --generate-and-test
echo.
echo Results will be saved in the 'test_results/' directory.
echo.
echo 📚 For more information, see README.md
echo.

echo [SUCCESS] Setup complete! 🎉
echo.
echo Press any key to exit...
pause >nul