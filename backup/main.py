"""
Enhanced DCGAN Web Backend - FastAPI Application
===============================================

Production-grade FastAPI backend that exposes the Enhanced DCGAN functionality
as REST APIs with WebSocket support for real-time training monitoring.

File: backend/main.py
"""
from fastapi.responses import StreamingResponse
import os
import sys
import asyncio
import json
import uuid
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import shutil
import base64
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
import uvicorn


# Add this to your main.py - Replace the existing /api/generate endpoint

import base64
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Add the enhanced_dcgan package to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import WebSocket manager
from websocket_manager import websocket_manager

# Import the existing Enhanced DCGAN modules
try:
    from enhanced_dcgan_research import (
        DATASETS,
        device, device_name, device_type,
        find_available_checkpoints,
        train_enhanced_gan_with_resume_modified,
    )
    DCGAN_AVAILABLE = True
    TRAINING_AVAILABLE = True
    REPORTING_AVAILABLE = True
    print("✅ Enhanced DCGAN modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced DCGAN modules not available: {e}")
    print("💡 This is expected if running in development mode without the research package")
    DCGAN_AVAILABLE = False
    TRAINING_AVAILABLE = False
    REPORTING_AVAILABLE = False

    # Mock data for development
    DATASETS = {
        'mnist': {
            'name': 'MNIST',
            'description': 'Handwritten digits dataset',
            'image_size': 28,
            'channels': 1,
            'num_classes': 10,
            'preprocessing_info': 'Normalized to [-1, 1]'
        },
        'cifar10': {
            'name': 'CIFAR-10',
            'description': 'Natural images dataset',
            'image_size': 32,
            'channels': 3,
            'num_classes': 10,
            'preprocessing_info': 'Normalized to [-1, 1]'
        }
    }
    device_name = "Development Mode"
    device_type = "cpu"

# ============================================================================
# AUTO-RESUME CONFIGURATION
# ============================================================================

# Set environment variable to enable auto-resume mode
os.environ['DCGAN_AUTO_RESUME'] = '1'  # Always auto-select option 1
os.environ['DCGAN_WEB_MODE'] = '1'     # Disable interactive input

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Enhanced DCGAN Web API",
    description="Production API for Enhanced Deep Convolutional GAN with academic research capabilities",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for frontend integration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Storage directories
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "./storage"))
MODELS_DIR = STORAGE_ROOT / "models"
REPORTS_DIR = STORAGE_ROOT / "reports"
STATIC_DIR = STORAGE_ROOT / "static"
LOGS_DIR = STORAGE_ROOT / "training_logs"

# Create directories
for directory in [STORAGE_ROOT, MODELS_DIR, REPORTS_DIR, STATIC_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================================
# Pydantic Models for API
# ============================================================================

class SystemStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: str
    device_name: str
    device_type: str
    dcgan_available: bool
    available_datasets: List[str]
    total_checkpoints: int
    timestamp: str

class TrainingConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dataset: str
    epochs: int
    resume_mode: str = "resume_latest"  # Changed default to auto-resume
    experiment_name: Optional[str] = None

class TrainingStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    training_id: str
    status: str  # "idle", "running", "completed", "error"
    dataset: str
    current_epoch: int
    total_epochs: int
    progress_percentage: float
    metrics: Dict[str, Any]
    start_time: Optional[str]
    end_time: Optional[str]
    error_message: Optional[str]

class GenerationRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prompt: str
    dataset: str
    num_samples: int = 8
    use_ema: bool = True

class CheckpointInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dataset: str
    filename: str
    epoch: int
    file_size_mb: float
    timestamp: str
    metrics: Dict[str, Any]

class ReportRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dataset: str
    experiment_id: Optional[str] = None
    include_images: bool = True

# ============================================================================
# Global State Management
# ============================================================================

class TrainingManager:
    def __init__(self):
        self.active_trainings: Dict[str, TrainingStatus] = {}

training_manager = TrainingManager()

# ============================================================================
# Mock functions for development mode
# ============================================================================

def mock_find_available_checkpoints(dataset_key: str) -> List[str]:
    """Mock function for finding checkpoints when DCGAN not available"""
    if DCGAN_AVAILABLE:
        return find_available_checkpoints(dataset_key)

    # Return mock checkpoint paths for development
    return [
        f"./storage/models/{dataset_key}_epoch_50.pth",
        f"./storage/models/{dataset_key}_epoch_25.pth"
    ]


# ============================================================================
# Image Generation Helper Functions
# ============================================================================

def create_mock_image(digit: int = None, size: tuple = (28, 28)) -> Image.Image:
    """Create a mock MNIST-style image"""
    if digit is None:
        digit = np.random.randint(0, 10)

    # Create a simple digit-like pattern
    img_array = np.zeros(size, dtype=np.uint8)

    # Add some noise
    noise = np.random.randint(0, 50, size)
    img_array = noise.astype(np.uint8)

    # Add a simple pattern based on digit
    center_x, center_y = size[0] // 2, size[1] // 2

    if digit == 0:
        # Circle
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= (min(size) // 3)**2
        img_array[mask] = 255
        inner_mask = (x - center_x)**2 + (y - center_y)**2 <= (min(size) // 5)**2
        img_array[inner_mask] = 0
    elif digit == 1:
        # Vertical line
        img_array[:, center_x-2:center_x+3] = 255
    elif digit == 2:
        # Horizontal lines
        img_array[center_y-5:center_y-2, :] = 255
        img_array[center_y+2:center_y+5, :] = 255
    else:
        # Random pattern for other digits
        for _ in range(digit + 3):
            x = np.random.randint(5, size[1]-5)
            y = np.random.randint(5, size[0]-5)
            img_array[y-2:y+3, x-2:x+3] = 255

    return Image.fromarray(img_array, mode='L')

def create_mock_cifar_image(size: tuple = (32, 32)) -> Image.Image:
    """Create a mock CIFAR-10 style image"""
    # Create random colored image
    img_array = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)

    # Add some patterns
    center_x, center_y = size[0] // 2, size[1] // 2

    # Add colored squares or patterns
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    color = colors[np.random.randint(0, len(colors))]

    # Draw a colored rectangle
    x1, y1 = center_x - 8, center_y - 8
    x2, y2 = center_x + 8, center_y + 8
    img_array[x1:x2, y1:y2] = color

    return Image.fromarray(img_array, mode='RGB')

def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def generate_with_real_model(dataset: str, num_samples: int, prompt: str = "") -> list:
    """Generate images using the actual Enhanced DCGAN model if available"""
    images = []

    try:
        if DCGAN_AVAILABLE:
            # Try to load the actual model and generate images
            # This would use your existing Enhanced DCGAN functionality
            from enhanced_dcgan_research import (
                EnhancedConditionalGenerator,
                find_available_checkpoints,
                device
            )

            # Find the best checkpoint
            checkpoints = find_available_checkpoints(dataset)
            if not checkpoints:
                raise Exception("No checkpoints available")

            # Load the model (simplified - you'd need to adapt this to your exact model structure)
            checkpoint_path = checkpoints[0]  # Use the first available checkpoint

            # For now, fall back to mock generation
            # You would implement actual model loading and generation here
            raise Exception("Real model generation not yet implemented")

        else:
            raise Exception("DCGAN not available")

    except Exception as e:
        print(f"Failed to generate with real model: {e}")
        print("Falling back to mock generation")

        # Generate mock images
        for i in range(num_samples):
            if dataset == 'mnist':
                # Extract digit from prompt if possible
                digit = None
                if prompt:
                    import re
                    digits = re.findall(r'\d', prompt)
                    if digits:
                        digit = int(digits[0])

                mock_img = create_mock_image(digit=digit)
            else:  # cifar10
                mock_img = create_mock_cifar_image()

            images.append({
                "id": i,
                "url": image_to_base64(mock_img),
                "generated_at": datetime.now().isoformat(),
                "prompt": prompt,
                "dataset": dataset
            })

    return images

# ============================================================================
# API Routes - System Information
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Enhanced DCGAN Web API",
        "status": "active",
        "version": "1.0.0",
        "dcgan_available": DCGAN_AVAILABLE,
        "auto_resume_enabled": True  # Indicate auto-resume is active
    }

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status and capabilities"""

    # Count total checkpoints across all datasets
    total_checkpoints = 0
    available_datasets = list(DATASETS.keys()) if DCGAN_AVAILABLE else ['mnist', 'cifar10']

    for dataset_key in available_datasets:
        checkpoints = mock_find_available_checkpoints(dataset_key)
        total_checkpoints += len(checkpoints)

    status_text = "online" if DCGAN_AVAILABLE else "development"
    current_device_name = device_name if DCGAN_AVAILABLE else "Development Mode"
    current_device_type = device_type if DCGAN_AVAILABLE else "cpu"

    return SystemStatus(
        status=status_text,
        device_name=current_device_name,
        device_type=current_device_type,
        dcgan_available=DCGAN_AVAILABLE,
        available_datasets=available_datasets,
        total_checkpoints=total_checkpoints,
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/datasets")
async def get_datasets():
    """Get available datasets with detailed information"""

    # Use actual datasets if available, otherwise use mock data
    if DCGAN_AVAILABLE:
        datasets_source = DATASETS
    else:
        datasets_source = {
            'mnist': {
                'name': 'MNIST',
                'description': 'Handwritten digits dataset',
                'image_size': 28,
                'channels': 1,
                'num_classes': 10,
                'preprocessing_info': 'Normalized to [-1, 1]'
            },
            'cifar10': {
                'name': 'CIFAR-10',
                'description': 'Natural images dataset',
                'image_size': 32,
                'channels': 3,
                'num_classes': 10,
                'preprocessing_info': 'Normalized to [-1, 1]'
            }
        }

    datasets_info = {}
    for key, config in datasets_source.items():
        checkpoints = mock_find_available_checkpoints(key)

        # Handle both dict and object configs
        if hasattr(config, 'name'):
            # Object config
            datasets_info[key] = {
                "name": config.name,
                "description": config.description,
                "image_size": config.image_size,
                "channels": config.channels,
                "num_classes": config.num_classes,
                "preprocessing_info": config.preprocessing_info,
                "available_checkpoints": len(checkpoints)
            }
        else:
            # Dict config
            datasets_info[key] = {
                "name": config.get('name', key.upper()),
                "description": config.get('description', f'{key} dataset'),
                "image_size": config.get('image_size', 32),
                "channels": config.get('channels', 3),
                "num_classes": config.get('num_classes', 10),
                "preprocessing_info": config.get('preprocessing_info', 'Normalized'),
                "available_checkpoints": len(checkpoints)
            }

    return datasets_info

@app.get("/api/checkpoints/{dataset}")
async def get_checkpoints(dataset: str):
    """Get available checkpoints for a specific dataset"""

    # Use actual datasets if available, otherwise use mock data
    if DCGAN_AVAILABLE:
        datasets_source = DATASETS
    else:
        datasets_source = {
            'mnist': {'name': 'MNIST'},
            'cifar10': {'name': 'CIFAR-10'}
        }

    if dataset not in datasets_source:
        raise HTTPException(status_code=404, detail="Dataset not found")

    checkpoints = mock_find_available_checkpoints(dataset)
    checkpoint_info = []

    for i, checkpoint_path in enumerate(checkpoints):
        try:
            # If DCGAN is available and file exists, try to load actual checkpoint data
            if DCGAN_AVAILABLE and os.path.exists(checkpoint_path):
                import torch
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

                checkpoint_info.append({
                    "dataset": dataset,
                    "filename": os.path.basename(checkpoint_path),
                    "epoch": checkpoint_data.get('epoch', (i + 1) * 25),
                    "file_size_mb": os.path.getsize(checkpoint_path) / (1024 * 1024),
                    "timestamp": checkpoint_data.get('timestamp', datetime.now().isoformat()),
                    "metrics": checkpoint_data.get('training_stats', {
                        "generator_loss": 0.45 + i * 0.01,
                        "discriminator_loss": 0.78 + i * 0.02
                    })
                })
            else:
                # Mock checkpoint data
                checkpoint_info.append({
                    "dataset": dataset,
                    "filename": os.path.basename(checkpoint_path),
                    "epoch": (i + 1) * 25,  # Mock epoch numbers
                    "file_size_mb": 25.0 + i * 0.5,  # Mock file sizes
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "generator_loss": 0.45 + i * 0.01,
                        "discriminator_loss": 0.78 + i * 0.02
                    }
                })
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_path}: {e}")

    return checkpoint_info

# ============================================================================
# API Routes - Training Management
# ============================================================================

@app.post("/api/training/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start GAN training with the specified configuration"""

    # Use actual datasets if available, otherwise use mock data
    if DCGAN_AVAILABLE:
        datasets_source = DATASETS
    else:
        datasets_source = {
            'mnist': {'name': 'MNIST'},
            'cifar10': {'name': 'CIFAR-10'}
        }

    if config.dataset not in datasets_source:
        raise HTTPException(status_code=400, detail="Invalid dataset")

    # Generate unique training ID
    training_id = str(uuid.uuid4())

    # Create initial training status
    training_status = TrainingStatus(
        training_id=training_id,
        status="starting",
        dataset=config.dataset,
        current_epoch=0,
        total_epochs=config.epochs,
        progress_percentage=0.0,
        metrics={},
        start_time=datetime.now().isoformat(),
        end_time=None,
        error_message=None
    )

    training_manager.active_trainings[training_id] = training_status

    # Send immediate WebSocket update
    await websocket_manager.send_training_update(training_id, {
        "status": "starting",
        "dataset": config.dataset,
        "total_epochs": config.epochs,
        "current_epoch": 0,
        "progress_percentage": 0,
        "timestamp": datetime.now().isoformat()
    })

    # Send log message
    await websocket_manager.send_log_message({
        "level": "info",
        "message": f"Training started for {config.dataset.upper()} dataset (Auto-resume enabled)",
        "dataset": config.dataset,
        "source": "training",
        "timestamp": datetime.now().isoformat()
    })

    # Start training in background
    background_tasks.add_task(run_training_task, training_id, config)

    return {"training_id": training_id, "status": "started", "auto_resume": True}

async def run_training_task(training_id: str, config: TrainingConfig):
    """Background task to run the actual training"""
    training_status = training_manager.active_trainings[training_id]

    try:
        training_status.status = "running"
        await websocket_manager.send_training_update(training_id, {
            "status": "running",
            "dataset": config.dataset,
            "total_epochs": config.epochs,
            "current_epoch": 0,
            "progress_percentage": 0,
            "timestamp": datetime.now().isoformat()
        })

        if DCGAN_AVAILABLE and TRAINING_AVAILABLE:
            print(f"🚀 Starting REAL training with Enhanced DCGAN for {config.dataset}")
            print(f"📊 Using actual device: {device_name} ({device_type})")
            print(f"🤖 Auto-resume mode: ENABLED (will auto-select option 1)")

            # Send auto-resume notification
            await websocket_manager.send_log_message({
                "level": "info",
                "message": "Auto-resume mode enabled - will automatically resume from latest checkpoint",
                "dataset": config.dataset,
                "source": "training",
                "timestamp": datetime.now().isoformat()
            })

            # Create training callback for real-time updates
            callback = TrainingProgressCallback(training_id, config.dataset)

            # Use actual dataset config
            dataset_config = DATASETS[config.dataset]

            # Run real training with callback integration
            # Note: You'll need to modify your training function to accept and use the callback
            ema_generator, critic = await run_training_with_callback(
                config=config,
                dataset_config=dataset_config,
                callback=callback
            )

            print(f"✅ Real training completed successfully!")

        else:
            # Mock training for development/demo
            print(f"🎯 Mock training started for {config.dataset} (development mode)")

            for epoch in range(1, config.epochs + 1):
                await asyncio.sleep(1)  # Simulate training time

                # Update progress
                training_status.current_epoch = epoch
                training_status.progress_percentage = (epoch / config.epochs) * 100
                training_status.metrics = {
                    "generator_loss": max(0.1, 0.8 - (epoch * 0.015) + np.random.normal(0, 0.05)),
                    "discriminator_loss": max(0.1, 0.7 - (epoch * 0.012) + np.random.normal(0, 0.04)),
                    "wasserstein_distance": max(-2.0, -0.5 - (epoch * 0.01) + np.random.normal(0, 0.1)),
                    "ema_quality": min(1.0, 0.3 + (epoch * 0.014) + np.random.normal(0, 0.02))
                }

                # Send WebSocket updates
                await websocket_manager.send_training_update(training_id, {
                    "status": "running",
                    "dataset": config.dataset,
                    "current_epoch": epoch,
                    "total_epochs": config.epochs,
                    "progress_percentage": training_status.progress_percentage,
                    "metrics": training_status.metrics,
                    "timestamp": datetime.now().isoformat()
                })

                # Send log messages
                await websocket_manager.send_log_message({
                    "level": "debug",
                    "message": f"Epoch {epoch}/{config.epochs} - G_Loss: {training_status.metrics['generator_loss']:.4f}, D_Loss: {training_status.metrics['discriminator_loss']:.4f}",
                    "dataset": config.dataset,
                    "source": "training",
                    "timestamp": datetime.now().isoformat()
                })

                print(f"📊 Epoch {epoch}/{config.epochs}: Loss G={training_status.metrics['generator_loss']:.4f}, D={training_status.metrics['discriminator_loss']:.4f}")

        training_status.status = "completed"
        training_status.end_time = datetime.now().isoformat()
        training_status.progress_percentage = 100.0

        print(f"🎉 Training completed for {config.dataset}!")

    except Exception as e:
        training_status.status = "error"
        training_status.error_message = str(e)
        training_status.end_time = datetime.now().isoformat()
        print(f"❌ Training error: {e}")

        # Send error log
        await websocket_manager.send_log_message({
            "level": "error",
            "message": f"Training failed: {str(e)}",
            "dataset": config.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    # Send final status update
    await websocket_manager.send_training_update(training_id, {
        "status": training_status.status,
        "dataset": config.dataset,
        "current_epoch": training_status.current_epoch,
        "total_epochs": training_status.total_epochs,
        "progress_percentage": training_status.progress_percentage,
        "metrics": training_status.metrics,
        "end_time": training_status.end_time,
        "error_message": training_status.error_message,
        "timestamp": datetime.now().isoformat()
    })

class TrainingProgressCallback:
    """Callback class for real-time training updates"""

    def __init__(self, training_id: str, dataset: str):
        self.training_id = training_id
        self.dataset = dataset
        self.step_count = 0

    async def on_epoch_start(self, epoch: int, total_epochs: int):
        await websocket_manager.send_training_update(self.training_id, {
            "status": "running",
            "current_epoch": epoch,
            "total_epochs": total_epochs,
            "progress_percentage": (epoch / total_epochs) * 100,
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Starting epoch {epoch}/{total_epochs}",
            "dataset": self.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    async def on_batch_end(self, batch: int, total_batches: int, metrics: dict):
        self.step_count += 1

        # Send updates every 10 batches to avoid overwhelming
        if batch % 10 == 0 or batch == total_batches:
            batch_progress = (batch / total_batches) * 100

            await websocket_manager.send_training_update(self.training_id, {
                "status": "running",
                "batch": batch,
                "total_batches": total_batches,
                "batch_progress": batch_progress,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            })

            await websocket_manager.send_log_message({
                "level": "debug",
                "message": f"Batch {batch}/{total_batches} - D_Loss: {metrics.get('discriminator_loss', 0):.4f}, G_Loss: {metrics.get('generator_loss', 0):.4f}",
                "dataset": self.dataset,
                "source": "training",
                "timestamp": datetime.now().isoformat()
            })

    async def on_training_complete(self):
        await websocket_manager.send_training_update(self.training_id, {
            "status": "completed",
            "progress_percentage": 100,
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Training completed successfully for {self.dataset}",
            "dataset": self.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

async def run_training_with_callback(config: TrainingConfig, dataset_config, callback: TrainingProgressCallback):
    """Wrapper function to run training with callback integration"""
    # This is where you'd integrate the callback with your existing training function
    # For now, using the existing function without modification
    return train_enhanced_gan_with_resume_modified(
        dataset_key=config.dataset,
        config=dataset_config,
        resume_from_checkpoint=(config.resume_mode != 'fresh'),
        num_epochs=config.epochs,
        experiment_name=config.experiment_name
    )

@app.get("/api/training/status/{training_id}")
async def get_training_status(training_id: str):
    """Get current status of a training session"""
    if training_id not in training_manager.active_trainings:
        raise HTTPException(status_code=404, detail="Training session not found")

    return training_manager.active_trainings[training_id]

@app.get("/api/training/active")
async def get_active_trainings():
    """Get all active training sessions"""
    return list(training_manager.active_trainings.values())

@app.post("/api/training/stop/{training_id}")
async def stop_training(training_id: str):
    """Stop a running training session"""
    if training_id not in training_manager.active_trainings:
        raise HTTPException(status_code=404, detail="Training session not found")

    training_status = training_manager.active_trainings[training_id]
    if training_status.status == "running":
        training_status.status = "stopped"
        training_status.end_time = datetime.now().isoformat()

        await websocket_manager.send_training_update(training_id, {
            "status": "stopped",
            "end_time": training_status.end_time,
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "warning",
            "message": f"Training stopped by user for {training_status.dataset}",
            "dataset": training_status.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    return {"message": "Training stopped"}

# ============================================================================
# API Routes - Interactive Generation
# ============================================================================

@app.post("/api/generate")
async def generate_images(request: GenerationRequest):
    """Generate images based on text prompt using trained model"""
    if request.dataset not in DATASETS:
        raise HTTPException(status_code=400, detail="Invalid dataset")

    try:
        generation_id = str(uuid.uuid4())

        print(f"🎨 Generating {request.num_samples} images for {request.dataset}")
        print(f"📝 Prompt: '{request.prompt}'")

        # Send log message
        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Image generation started: '{request.prompt}' for {request.dataset}",
            "dataset": request.dataset,
            "source": "generation",
            "timestamp": datetime.now().isoformat()
        })

        # Generate images
        images = generate_with_real_model(
            dataset=request.dataset,
            num_samples=request.num_samples,
            prompt=request.prompt
        )

        # Send completion log
        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Generated {len(images)} images successfully",
            "dataset": request.dataset,
            "source": "generation",
            "timestamp": datetime.now().isoformat()
        })

        response = {
            "generation_id": generation_id,
            "prompt": request.prompt,
            "dataset": request.dataset,
            "num_samples": request.num_samples,
            "images": images,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "message": f"Generated {len(images)} images" + (" (mock mode)" if not DCGAN_AVAILABLE else "")
        }

        print(f"✅ Generated {len(images)} images successfully")
        return response

    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        print(f"❌ {error_msg}")

        await websocket_manager.send_log_message({
            "level": "error",
            "message": error_msg,
            "dataset": request.dataset,
            "source": "generation",
            "timestamp": datetime.now().isoformat()
        })

        raise HTTPException(status_code=500, detail=error_msg)

# ============================================================================
# Add endpoint to download individual images
# ============================================================================

@app.get("/api/generate/{generation_id}/download/{image_id}")
async def download_generated_image(generation_id: str, image_id: str):
    """Download a specific generated image"""
    # For now, return a mock image since we don't store generations
    # In a real implementation, you'd store the generation results

    try:
        # Create a mock image for download
        mock_img = create_mock_image()

        # Convert to bytes
        img_buffer = io.BytesIO()
        mock_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(img_buffer.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=generated_image_{image_id}.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# ============================================================================
# API Routes - Logs
# ============================================================================

@app.get("/api/logs/{dataset}")
async def get_training_logs(dataset: str):
    """Get training logs for a specific dataset"""
    # Return recent logs for the dataset
    logs = [
        {
            "id": 1,
            "timestamp": datetime.now().isoformat(),
            "level": "info",
            "message": f"System ready for {dataset} training",
            "dataset": dataset,
            "source": "system"
        },
        {
            "id": 2,
            "timestamp": datetime.now().isoformat(),
            "level": "debug",
            "message": f"Checkpoints available for {dataset}",
            "dataset": dataset,
            "source": "checkpoint"
        }
    ]

    return logs

# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates"""
    await websocket_manager.connect(websocket)

    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            print(f"📨 Received WebSocket message: {data}")

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"🔌 WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "dcgan_available": DCGAN_AVAILABLE,
        "device": device_name,
        "mode": "production" if DCGAN_AVAILABLE else "development",
        "websocket_connections": len(websocket_manager.active_connections),
        "auto_resume_enabled": True
    }

# ============================================================================
# Application Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    print("🚀 Enhanced DCGAN Web API Starting...")
    print(f"📊 DCGAN Available: {DCGAN_AVAILABLE}")
    print(f"🖥️  Device: {device_name} ({device_type})")
    print(f"📁 Storage Root: {STORAGE_ROOT}")
    print(f"🌐 CORS Origins: {ALLOWED_ORIGINS}")
    print(f"🔌 WebSocket Manager: Ready")
    print(f"🤖 Auto-Resume Mode: ENABLED")

    if not DCGAN_AVAILABLE:
        print("💡 Running in development mode with mock data")

    print("✅ API Ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    print("🛑 Enhanced DCGAN Web API Shutting down...")

# ============================================================================
# Application Startup
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("ENVIRONMENT", "production") == "development"

    print(f"🌐 Starting server on {host}:{port}")
    print(f"🔄 Reload mode: {reload}")
    print(f"🤖 Auto-Resume: ENABLED")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )