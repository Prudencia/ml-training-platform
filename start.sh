#!/bin/bash

# ML Training Platform - Startup Script

echo "Starting ML Training Platform..."

# Start backend
echo "Starting backend server..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 3

# Start frontend
echo "Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================"
echo "ML Training Platform is running!"
echo "========================================"
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "========================================"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
