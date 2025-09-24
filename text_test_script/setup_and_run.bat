@echo off
REM AI Detector API Testing - Easy Setup Script for Windows
REM This script automatically sets up everything needed to run the basic API testing tool

setlocal enabledelayedexpansion

echo.
echo ==================================================
echo   AI Detector API Testing - Easy Setup
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
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install some dependencies
    echo [ERROR] Try running manually: pip install -r requirements.txt
    pause
    exit /b 1
) else (
    echo [SUCCESS] All dependencies installed successfully
)

REM Create .env file for API configuration
echo [INFO] Setting up API configuration...
if exist ".env" (
    echo [WARNING] .env file already exists. Using existing configuration...
) else (
    echo [INFO] Creating .env file for API configuration...
    (
    echo # AI Detector API Configuration
    echo DETECTOR_API_URL=http://localhost:8000
    echo DETECTOR_API_TOKEN=
    echo.
    echo # Optional: Custom output directory
    echo OUTPUT_DIR=test_results
    ) > .env
    echo [SUCCESS] .env file created
)

echo.
echo ==================================================
echo   🔧 API CONFIGURATION
echo ==================================================
echo.
echo Default settings created in .env file:
echo - API URL: http://localhost:8000 (local detector^)
echo - No authentication token (for local testing^)
echo.
echo If your detector API is running elsewhere, edit the .env file:
echo - DETECTOR_API_URL=http://your-detector-url:8000
echo - DETECTOR_API_TOKEN=your-auth-token (if needed^)
echo.

REM Check sample data
echo [INFO] Checking sample test data...
if exist "sample_test_data.csv" (
    for /f %%i in ('type "sample_test_data.csv" ^| find /c /v ""') do set LINE_COUNT=%%i
    set /a SAMPLE_COUNT=!LINE_COUNT!-1
    echo [SUCCESS] Found sample data with approximately !SAMPLE_COUNT! test samples
) else (
    echo [WARNING] No sample_test_data.csv found
    echo [INFO] You can create your own CSV file with columns:
    echo         text,label,model_name,word_count,sentence_count,domain,source,confidence,language
)

REM Create a simple run script for Windows
echo [INFO] Creating easy run script...
(
echo @echo off
echo REM Simple test runner - activates venv and runs the basic API test
echo.
echo echo 🧪 Starting AI Detector API Test...
echo echo.
echo.
echo REM Activate virtual environment
echo call venv\Scripts\activate.bat
echo.
echo REM Check if sample data exists
echo if not exist "sample_test_data.csv" (
echo     echo ⚠️  No sample_test_data.csv found. Please add your test data.
echo     echo Expected CSV format: text,label,model_name,word_count,sentence_count,domain,source,confidence,language
echo     echo.
echo     echo Available CSV files:
echo     dir /b *.csv 2^>nul ^|^| echo No CSV files found in current directory
echo     echo.
echo     set /p CSV_FILE="Enter CSV filename to test: "
echo     if not exist "%%CSV_FILE%%" (
echo         echo File not found: %%CSV_FILE%%
echo         pause
echo         exit /b 1
echo     ^)
echo ^) else (
echo     set CSV_FILE=sample_test_data.csv
echo ^)
echo.
echo REM Run the test
echo echo Testing with file: %%CSV_FILE%%
echo python test_detector_api.py --input "%%CSV_FILE%%" --detailed
echo.
echo echo.
echo echo ✅ Test completed! Check the test_results/ directory for detailed reports.
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
echo 1. Double-click run_test.bat (easiest way^)
echo.
echo 2. Or use Command Prompt:
echo    run_test.bat
echo.
echo 3. Manual commands:
echo    venv\Scripts\activate.bat
echo    python test_detector_api.py --input sample_test_data.csv
echo.
echo Results will be saved in the 'test_results/' directory.
echo.
echo 📚 For more information, see README.md
echo.

echo [SUCCESS] Setup complete! 🎉
echo.
echo Press any key to exit...
pause >nul