#!/bin/bash
# Oelala Web Interface Startup Script

echo "🎬 Starting Oelala AI Video Generator Web Interface"
echo "=================================================="

# Check if we're in the right directory
if [ ! -d "/home/flip/oelala" ]; then
    echo "❌ Error: Oelala directory not found at /home/flip/oelala"
    exit 1
fi

cd /home/flip/oelala

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

echo "🚀 Starting FastAPI backend..."
cd src/backend
source /home/flip/venvs/gpu/bin/activate

# Start backend in background
python app.py &
BACKEND_PID=$!

echo "✅ Backend started (PID: $BACKEND_PID)"
echo "🌐 Backend URL: http://192.168.1.2:7998"
echo "📊 API Docs: http://192.168.1.2:7998/docs"

# Wait a moment for backend to start
sleep 3

echo ""
echo "🎨 Starting React frontend..."
cd ../frontend

# Start frontend in background
npm run dev &
FRONTEND_PID=$!

echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo "🌐 Frontend URL: http://192.168.1.2:5174"

echo ""
echo "🎉 Oelala Web Interface is running!"
echo "📱 Open http://192.168.1.2:5174 in your browser"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait
