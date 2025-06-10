#!/bin/bash
# Enhanced DCGAN Backend Startup Script
# File: backend/start_backend.sh

echo "🚀 Starting Enhanced DCGAN Backend with WebSocket Support (v02)..."

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  No virtual environment detected"
    echo "💡 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
else
    echo "✅ Virtual environment detected"
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
fi

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
    echo "⚠️  Python 3.8+ is recommended for this application"
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "📦 Creating basic requirements.txt..."
    cat > requirements.txt << EOF
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
websockets>=12.0
pydantic>=2.5.0
python-multipart>=0.0.6
numpy>=1.24.0
python-dotenv>=1.0.0
torch>=2.0.0
torchvision>=0.15.0
requests>=2.31.0
aiofiles>=23.0.0
EOF
    echo "✅ Basic requirements.txt created"
fi

# Check if critical packages are installed
MISSING_PACKAGES=()

echo "🔍 Checking critical dependencies..."

# Check FastAPI
if ! python -c "import fastapi" 2>/dev/null; then
    MISSING_PACKAGES+=("fastapi")
fi

# Check uvicorn
if ! python -c "import uvicorn" 2>/dev/null; then
    MISSING_PACKAGES+=("uvicorn")
fi

# Check websockets
if ! python -c "import websockets" 2>/dev/null; then
    MISSING_PACKAGES+=("websockets")
fi

# Check pydantic
if ! python -c "import pydantic" 2>/dev/null; then
    MISSING_PACKAGES+=("pydantic")
fi

# Install missing packages
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "📦 Installing missing packages: ${MISSING_PACKAGES[*]}"
    pip install -r requirements.txt
    echo "✅ Requirements installed"
else
    echo "✅ All critical dependencies are installed"
fi

# Check for Enhanced DCGAN package
echo "🔍 Checking Enhanced DCGAN package..."
if python -c "import enhanced_dcgan_research" 2>/dev/null; then
    echo "✅ Enhanced DCGAN package available"
    DCGAN_MODE="production"
else
    echo "⚠️  Enhanced DCGAN package not found"
    echo "💡 Running in development mode with mock data"
    DCGAN_MODE="development"
fi

# Create storage directories
echo "📁 Creating storage directories..."
mkdir -p storage/{models,reports,static,training_logs}
mkdir -p storage/models/{mnist,cifar10}
mkdir -p storage/models/mnist/enhanced
mkdir -p storage/models/mnist/emergency
mkdir -p storage/reports/{mnist,cifar10}
echo "✅ Storage directories created"

# Load environment variables
if [ -f ".env" ]; then
    echo "🔧 Loading environment variables from .env"
    set -a  # Automatically export all variables
    source .env
    set +a  # Stop auto-exporting
else
    echo "⚠️  No .env file found, creating basic .env..."
    cat > .env << EOF
# Enhanced DCGAN Backend Configuration
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=$DCGAN_MODE
STORAGE_ROOT=./storage
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Training Configuration
DEFAULT_BATCH_SIZE=64
DEFAULT_EPOCHS=50

# Logging
LOG_LEVEL=INFO
EOF
    echo "✅ Basic .env file created"
    set -a
    source .env
    set +a
fi

# Set default values if not provided
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}
export ENVIRONMENT=${ENVIRONMENT:-$DCGAN_MODE}
export STORAGE_ROOT=${STORAGE_ROOT:-"./storage"}

# Check if port is in use and find alternative
ORIGINAL_PORT=$PORT
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    echo "⚠️  Port $PORT is already in use"
    PORT=$((PORT + 1))
    echo "🔄 Trying port $PORT"
done

if [ $PORT != $ORIGINAL_PORT ]; then
    echo "📝 Port changed from $ORIGINAL_PORT to $PORT"
    export PORT=$PORT
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory"
    echo "💡 Make sure you're running this script from the backend directory"
    exit 1
fi

# Check if websocket_manager.py exists
if [ ! -f "websocket_manager.py" ]; then
    echo "⚠️  websocket_manager.py not found"
    echo "💡 WebSocket functionality may not work properly"
    echo "📝 Please ensure websocket_manager.py is in the backend directory"
fi

# Display startup information
echo ""
echo "🔧 STARTUP CONFIGURATION"
echo "=========================="
echo "🌐 Host: $HOST"
echo "🔌 Port: $PORT"
echo "🏗️  Mode: $ENVIRONMENT"
echo "📁 Storage: $STORAGE_ROOT"
echo "🔗 CORS Origins: $ALLOWED_ORIGINS"
echo "🖥️  Python: $(which python)"
echo "📦 Package Mode: $DCGAN_MODE"
echo ""

# Health check function
check_server_health() {
    echo "🏥 Performing health check..."

    # Wait a moment for server to start
    sleep 3

    # Try to connect to the health endpoint
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✅ Server health check passed"
        echo "📍 Server is running at: http://localhost:$PORT"
        echo "📖 API documentation: http://localhost:$PORT/api/docs"
        echo "🔌 WebSocket endpoint: ws://localhost:$PORT/ws"
    else
        echo "❌ Server health check failed"
        echo "💡 Check the server logs above for errors"
    fi
}

# Function to handle shutdown gracefully
cleanup() {
    echo ""
    echo "🛑 Shutting down Enhanced DCGAN Backend..."
    echo "💾 Saving any pending data..."

    # Kill any remaining processes
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null
    fi

    echo "✅ Shutdown complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the backend server
echo "🌐 Starting FastAPI server with WebSocket support..."
echo "🛑 Press Ctrl+C to stop the server"
echo "📊 Mode: $DCGAN_MODE"
echo ""

# Start server and capture PID
if [ "$ENVIRONMENT" = "development" ]; then
    echo "🔄 Starting in development mode with auto-reload..."
    python -c "
import uvicorn
import os

if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='$HOST',
        port=$PORT,
        reload=True,
        log_level='info',
        access_log=True
    )
" &
else
    echo "🚀 Starting in production mode..."
    python -c "
import uvicorn
import os

if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='$HOST',
        port=$PORT,
        reload=False,
        log_level='info',
        access_log=True,
        workers=1
    )
" &
fi

SERVER_PID=$!

# Run health check in background
(check_server_health) &

# Wait for the server process
wait $SERVER_PID