@echo off
cls
echo.
echo ============================================================
echo     SENTRAVISION - WEB SERVER LAUNCHER
echo ============================================================
echo.

cd /d "%~dp0"

echo [Step 1/3] Checking for existing server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3000" ^| findstr "LISTENING"') do (
    echo    ^> Stopping old server on port 3000...
    taskkill /F /PID %%a >nul 2>&1
)

echo [Step 2/3] Starting SentraVision server...
cd WebApp
start /B node server.js

timeout /t 2 /nobreak >nul

echo [Step 3/3] Opening browser...
start "" "http://localhost:3000"

echo.
echo ============================================================
echo     SERVER IS RUNNING!
echo ============================================================
echo.
echo     Open in browser: http://localhost:3000
echo.
echo     Press Ctrl+C in this window to stop the server
echo ============================================================
echo.

node server.js
