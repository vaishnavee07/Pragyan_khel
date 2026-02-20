@echo off
echo Starting SentraVision Backend...
cd backend
call venv\Scripts\activate
start cmd /k python main.py

timeout /t 3 /nobreak > nul

echo Starting SentraVision Frontend...
cd ..\frontend
start cmd /k npm run dev

timeout /t 5 /nobreak > nul

echo Opening browser...
start http://localhost:3000

echo SentraVision started!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
