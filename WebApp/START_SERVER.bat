@echo off
echo ============================================================
echo     SENTRAVISION WEB SERVER - STARTING
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/2] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [2/2] Starting web server on port 8000...
echo.
echo ============================================================
echo     SERVER IS RUNNING
echo ============================================================
echo.
echo     Open your browser and visit:
echo     http://localhost:8000
echo.
echo     Press Ctrl+C to stop the server
echo ============================================================
echo.

python -m http.server 8000
