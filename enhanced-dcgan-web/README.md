# Enhanced DCGAN Web Application

<div align="center">

![Enhanced DCGAN](https://img.shields.io/badge/Enhanced-DCGAN-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-00a8ff?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Production-ready web interface for Enhanced Deep Convolutional Generative Adversarial Networks**

[🚀 Quick Start](#quick-start) • [📖 Documentation](#documentation) • [🐛 Issues](https://github.com/jahidul-arafat/gan-mnist-cifar/issues) • [💡 Features](#features)

</div>

## 🎯 Overview

Enhanced DCGAN Web Application provides a professional web interface for training and deploying Deep Convolutional GANs with advanced features including real-time monitoring, interactive generation, and academic reporting capabilities.

### 🏗️ Architecture

```
enhanced-dcgan-web/
├── 📁 backend/                    # FastAPI Backend
│   ├── main.py                    # Main API application  
│   ├── requirements.txt           # Python dependencies
│   └── health_check.py           # Health monitoring
├── 📁 frontend/                   # React Frontend
│   ├── 📁 src/components/         # React components
│   ├── 📁 src/services/          # API integration
│   └── 📁 src/hooks/             # Custom React hooks
├── 📁 deployment/                 # Deployment configs
│   └── render.yaml               # Render deployment
└── 📁 scripts/                   # Utility scripts
```

## ✨ Features

### 🔬 Core DCGAN Features
- **WGAN-GP Loss** - Wasserstein GAN with Gradient Penalty for stable training
- **EMA Generator** - Exponential Moving Average for improved sample quality
- **Enhanced Architecture** - Advanced convolutional layers with spectral normalization
- **Multi-Device Support** - CUDA, Apple Metal MPS, and CPU acceleration
- **Checkpoint Management** - Auto-save every 5 epochs with resume capability

### 🌐 Web Interface Features
- **Real-time Training Monitoring** - Live progress tracking with WebSocket updates
- **Interactive Generation** - Natural language prompts like "Draw me a 7"
- **Academic Reporting** - Comprehensive research reports with generated images
- **Dashboard Analytics** - System status, training metrics, and performance insights
- **Responsive Design** - Modern UI with dark/light theme support

### 🚀 Production Features
- **Docker Support** - Containerized deployment with docker-compose
- **Cloud Ready** - Optimized for Render, AWS, GCP, and Azure
- **Health Monitoring** - Comprehensive health checks and system diagnostics
- **API Documentation** - Auto-generated OpenAPI/Swagger docs
- **Error Recovery** - Graceful error handling and automatic restart

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- 8GB+ RAM recommended
- GPU optional (CUDA/MPS supported)

### Local Development

1. **Clone and Setup**
   ```bash
   git clone https://github.com/your-username/enhanced-dcgan-web.git
   cd enhanced-dcgan-web
   cp .env.example .env
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   python health_check.py  # Verify installation
   uvicorn main:app --reload --port 8000
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. **Access Application**
    - Frontend: http://localhost:3000
    - Backend API: http://localhost:8000
    - API Docs: http://localhost:8000/api/docs

### 🐳 Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Or run individual services
docker build -f deployment/Dockerfile.backend -t dcgan-backend .
docker build -f deployment/Dockerfile.frontend -t dcgan-frontend .
```

### ☁️ Cloud Deployment (Render)

1. **Fork Repository**
2. **Connect to Render**
    - Import your repository
    - Use `deployment/render.yaml` for configuration
3. **Environment Variables**
    - Copy values from `.env.example`
    - Set production secrets
4. **Deploy**
    - Automatic deployment on git push
    - Monitor via Render dashboard

## 📖 Documentation

### 🎮 Usage Guide

#### Training a Model
1. Navigate to **Training Interface**
2. Select dataset (MNIST/CIFAR-10)
3. Configure epochs and resume mode
4. Click **Start Training**
5. Monitor real-time progress

#### Interactive Generation
1. Go to **Generate** tab
2. Enter natural language prompts:
    - "Draw me a 7"
    - "Generate cat"
    - "Show me a dog"
3. View generated images instantly

#### Academic Reports
1. Visit **Reports** panel
2. Select dataset and experiment
3. Generate comprehensive research reports
4. Download reports with embedded images

### 🔧 API Reference

#### Core Endpoints
```bash
# System Status
GET /api/system/status

# Start Training  
POST /api/training/start
{
  "dataset": "mnist",
  "epochs": 50,
  "resume_mode": "fresh"
}

# Interactive Generation
POST /api/generate
{
  "prompt": "Draw me a 7",
  "dataset": "mnist", 
  "num_samples": 8
}

# Generate Report
POST /api/reports/generate
{
  "dataset": "mnist",
  "include_images": true
}
```

#### WebSocket Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'training_status') {
    // Handle real-time training updates
  }
};
```

### 🎨 Component Architecture

#### Frontend Structure
```jsx
// Main App Component
<App>
  <Sidebar>
    <Dashboard />      // System overview
    <TrainingInterface />  // Training controls
    <InteractiveGeneration />  // Image generation
    <AnalyticsPanel />     // Metrics and charts
    <ReportsPanel />       // Academic reports
    <LogsPanel />         // System logs
  </Sidebar>
</App>
```

#### Custom Hooks
```javascript
// System status monitoring
const { systemStatus, isLoading } = useSystemStatus();

// Training session management  
const { activeTrainings, addTraining } = useTrainingStatus();

// Real-time WebSocket connection
const { isConnected, lastMessage } = useWebSocket();
```

## 🔧 Configuration

### Environment Variables

#### Backend Configuration
```bash
# Core Settings
ENVIRONMENT=production
SECRET_KEY=your-secret-key
PORT=8000

# DCGAN Settings  
DEFAULT_EPOCHS=50
MAX_CONCURRENT_TRAININGS=2
CUDA_VISIBLE_DEVICES=0

# Storage
STORAGE_ROOT=./storage
MODELS_DIR=./storage/models
```

#### Frontend Configuration
```bash
# API Connection
REACT_APP_API_URL=http://localhost:8000

# Build Settings
GENERATE_SOURCEMAP=false
CI=true
```

### Hardware Optimization

#### GPU Configuration
```bash
# NVIDIA CUDA
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Apple Metal MPS  
PYTORCH_ENABLE_MPS_FALLBACK=1

# CPU Optimization
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

## 🚢 Deployment

### Production Checklist

- [ ] Set secure `SECRET_KEY`
- [ ] Configure `ALLOWED_ORIGINS`
- [ ] Set up SSL certificates
- [ ] Configure monitoring
- [ ] Set resource limits
- [ ] Enable logging
- [ ] Configure backups

### Deployment Platforms

#### Render.com
```yaml
# deployment/render.yaml
services:
  - name: enhanced-dcgan-backend
    type: web
    env: python
    buildCommand: pip install -r backend/requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

#### Docker

```dockerfile
# Multi-stage build for production
FROM python:3.11-slim as backend
WORKDIR /app
COPY ../backup/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

#### AWS/G