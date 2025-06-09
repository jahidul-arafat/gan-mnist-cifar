#!/bin/bash
# Enhanced DCGAN Backend Startup Script
# File: backend/start_backend.sh

echo "🚀 Starting Enhanced DCGAN Backend..."

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

# Check if requirements are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
    echo "✅ Requirements installed"
else
    echo "✅ Requirements already installed"
fi

# Create storage directories
echo "📁 Creating storage directories..."
mkdir -p storage/{models,reports,static,training_logs}
echo "✅ Storage directories created"

# Load environment variables
if [ -f ".env" ]; then
    echo "🔧 Loading environment variables from .env"
    export $(cat .env | grep -v ^# | xargs)
else
    echo "⚠️  No .env file found, using defaults"
fi

# Check if port is in use and find alternative
PORT=${PORT:-8000}
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; do
    echo "⚠️  Port $PORT is already in use"
    PORT=$((PORT + 1))
    echo "🔄 Trying port $PORT"
done

export PORT=$PORT

# Start the backend server
echo "🌐 Starting FastAPI server..."
echo "📍 Server will be available at: http://localhost:$PORT"
echo "📖 API documentation at: http://localhost:$PORT/api/docs"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python main.py