#!/bin/bash

echo "Starting SentraVision Backend..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!

sleep 3

echo "Starting SentraVision Frontend..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

sleep 5

echo "Opening browser..."
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:3000
elif command -v open > /dev/null; then
    open http://localhost:3000
fi

echo "SentraVision started!"
echo "Backend: http://localhost:8000 (PID: $BACKEND_PID)"
echo "Frontend: http://localhost:3000 (PID: $FRONTEND_PID)"
echo ""
echo "Press Ctrl+C to stop both servers"

wait
