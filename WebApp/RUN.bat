@echo off
echo ============================================================
echo     SENTRAVISION - QUICK START
echo ============================================================
echo.

cd /d "%~dp0"

echo Starting server...
start "" START_SERVER.bat

timeout /t 3 /nobreak >nul

echo Opening browser...
start "" "http://localhost:8000"

echo.
echo Server started! Browser should open automatically.
echo.
echo If browser doesn't open, manually visit: http://localhost:8000
echo.
pause
