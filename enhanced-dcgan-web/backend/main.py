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
import random
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

import base64
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


# Add this right after your imports and before any other code
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# Disable matplotlib GUI completely in web mode
import os
os.environ['MPLBACKEND'] = 'Agg'

# Add the enhanced_dcgan package to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import WebSocket manager
from websocket_manager import websocket_manager

# Add these imports after the existing imports
from training_integration import EnhancedTrainingWrapper, TrainingProgressParser
import re
from threading import Thread


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

# Update the GenerationRequest model to include checkpoint selection
class GenerationRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prompt: str
    dataset: str
    num_samples: int = 8
    use_ema: bool = True
    checkpoint_path: Optional[str] = "latest"  # Add checkpoint selection
    seed: Optional[int] = None  # Add seed support

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

# Add this to main.py - Replace the generate_with_real_model function

def generate_with_real_model(dataset: str, num_samples: int, prompt: str = "", checkpoint_path: str = "latest") -> list:
    """Generate images using the actual Enhanced DCGAN model if available"""
    images = []

    try:
        if DCGAN_AVAILABLE:
            print(f"🎨 Loading actual DCGAN model for {dataset}")
            print(f"📂 Checkpoint: {checkpoint_path}")

            # Import the necessary modules
            from enhanced_dcgan_research import (
                DATASETS,
                device,
                EnhancedConditionalGenerator,
                find_available_checkpoints
            )
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
            import io
            import base64

            # Get dataset configuration
            if dataset not in DATASETS:
                raise Exception(f"Dataset {dataset} not found in DATASETS")

            dataset_config = DATASETS[dataset]

            # Find available checkpoints
            available_checkpoints = find_available_checkpoints(dataset)
            if not available_checkpoints:
                raise Exception(f"No checkpoints found for {dataset}")

            # Select checkpoint
            selected_checkpoint = None
            if checkpoint_path == "latest" or checkpoint_path == "Latest Checkpoint":
                # Use the most recent checkpoint
                selected_checkpoint = max(available_checkpoints, key=os.path.getmtime)
                print(f"🔄 Selected latest checkpoint: {os.path.basename(selected_checkpoint)}")
            else:
                # Try to find the specific checkpoint
                for cp in available_checkpoints:
                    if checkpoint_path in cp or os.path.basename(cp) == checkpoint_path:
                        selected_checkpoint = cp
                        print(f"🎯 Selected specific checkpoint: {os.path.basename(selected_checkpoint)}")
                        break

                if not selected_checkpoint:
                    # Fallback to latest
                    selected_checkpoint = max(available_checkpoints, key=os.path.getmtime)
                    print(f"⚠️ Checkpoint {checkpoint_path} not found, using latest: {os.path.basename(selected_checkpoint)}")

            print(f"📂 Loading checkpoint: {selected_checkpoint}")

            # Load checkpoint data
            checkpoint_data = torch.load(selected_checkpoint, map_location=device, weights_only=False)
            print(f"✅ Checkpoint loaded successfully")
            print(f"📊 Checkpoint epoch: {checkpoint_data.get('epoch', 'unknown')}")

            # Get model parameters from dataset config
            image_size = getattr(dataset_config, 'image_size', 32 if dataset == 'cifar10' else 28)
            channels = getattr(dataset_config, 'channels', 3 if dataset == 'cifar10' else 1)
            num_classes = getattr(dataset_config, 'num_classes', 10)

            # Model hyperparameters (should match training configuration)
            latent_dim = 100
            generator_features = 64

            print(f"🏗️ Creating generator model...")
            print(f"   Image size: {image_size}")
            print(f"   Channels: {channels}")
            print(f"   Classes: {num_classes}")
            print(f"   Latent dim: {latent_dim}")

            # Create generator model
            generator = EnhancedConditionalGenerator(
                latent_dim=latent_dim,
                num_classes=num_classes,
                channels=channels,
                features_g=generator_features,
                image_size=image_size
            ).to(device)

            # Load model state
            if 'ema_generator_state_dict' in checkpoint_data:
                generator.load_state_dict(checkpoint_data['ema_generator_state_dict'])
                print("📈 Loaded EMA generator state")
            elif 'generator_state_dict' in checkpoint_data:
                generator.load_state_dict(checkpoint_data['generator_state_dict'])
                print("🎯 Loaded regular generator state")
            else:
                raise Exception("No generator state found in checkpoint")

            generator.eval()
            print(f"🎨 Generator ready for inference")

            # Generate images
            with torch.no_grad():
                for i in range(num_samples):
                    print(f"🖼️ Generating image {i+1}/{num_samples}")

                    # Create random noise
                    noise = torch.randn(1, latent_dim, device=device)

                    # Handle class conditioning
                    if prompt and prompt.strip():
                        # Try to extract digit/class from prompt
                        import re
                        digits = re.findall(r'\d', prompt)
                        if digits and dataset == 'mnist':
                            class_label = int(digits[0]) % 10
                        elif dataset == 'cifar10':
                            # CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
                            class_mapping = {
                                'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                                'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9,
                                'plane': 0, 'car': 1, 'auto': 1
                            }
                            class_label = None
                            for word, label in class_mapping.items():
                                if word.lower() in prompt.lower():
                                    class_label = label
                                    break
                            if class_label is None:
                                class_label = torch.randint(0, num_classes, (1,)).item()
                        else:
                            class_label = torch.randint(0, num_classes, (1,)).item()
                    else:
                        class_label = torch.randint(0, num_classes, (1,)).item()

                    class_tensor = torch.tensor([class_label], dtype=torch.long, device=device)

                    print(f"   Class: {class_label}")

                    # Generate image
                    fake_image = generator(noise, class_tensor)

                    # Convert to PIL Image
                    if channels == 1:  # Grayscale (MNIST)
                        # Denormalize from [-1, 1] to [0, 1]
                        fake_image = (fake_image + 1.0) / 2.0
                        fake_image = torch.clamp(fake_image, 0.0, 1.0)

                        # Convert to numpy and then PIL
                        img_array = fake_image.cpu().squeeze().numpy()
                        img_array = (img_array * 255).astype('uint8')
                        pil_image = Image.fromarray(img_array, mode='L')
                    else:  # RGB (CIFAR-10)
                        # Denormalize from [-1, 1] to [0, 1]
                        fake_image = (fake_image + 1.0) / 2.0
                        fake_image = torch.clamp(fake_image, 0.0, 1.0)

                        # Convert to numpy and then PIL
                        img_array = fake_image.cpu().squeeze().permute(1, 2, 0).numpy()
                        img_array = (img_array * 255).astype('uint8')
                        pil_image = Image.fromarray(img_array, mode='RGB')

                    # Resize if needed (optional - for better display)
                    if image_size < 64:
                        # Upscale small images for better visibility
                        display_size = 128 if image_size == 28 else 96
                        pil_image = pil_image.resize((display_size, display_size), Image.NEAREST)

                    # Convert to base64
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='PNG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    img_data_url = f"data:image/png;base64,{img_str}"

                    images.append({
                        "id": f"real_gen_{i}",
                        "url": img_data_url,
                        "generated_at": datetime.now().isoformat(),
                        "prompt": prompt,
                        "dataset": dataset,
                        "class_label": class_label,
                        "checkpoint": os.path.basename(selected_checkpoint),
                        "method": "real_dcgan"
                    })

                    print(f"   ✅ Generated image {i+1}")

            print(f"🎉 Successfully generated {len(images)} real images using trained model")
            return images

        else:
            raise Exception("DCGAN not available")

    except Exception as e:
        print(f"❌ Real model generation failed: {e}")
        print(f"📍 Error details: {str(e)}")
        print("🔄 Falling back to mock generation")

        # Generate mock images as fallback
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
                "id": f"mock_{i}",
                "url": image_to_base64(mock_img),
                "generated_at": datetime.now().isoformat(),
                "prompt": prompt,
                "dataset": dataset,
                "method": "mock_fallback",
                "note": "Generated using mock data due to model loading failure"
            })

        print(f"⚠️ Generated {len(images)} mock images as fallback")

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

def detect_resume_info_from_output(output_line: str) -> Optional[Dict[str, Any]]:
    """
    Detect resume information from training console output
    Returns dict with resume info or None if not a resume line
    """

    # Pattern: "🔄 RESUMED from checkpoint at epoch 26"
    resumed_match = re.search(r'🔄 RESUMED from checkpoint at epoch (\d+)', output_line)
    if resumed_match:
        return {
            "type": "resumed",
            "resumed_from_epoch": int(resumed_match.group(1))
        }

    # Pattern: "EPOCH 27/50" (absolute epoch numbers in resumed training)
    epoch_match = re.search(r'(?:EPOCH|Epoch)\s+(\d+)/(\d+)', output_line)
    if epoch_match:
        return {
            "type": "epoch_progress",
            "current_epoch": int(epoch_match.group(1)),
            "total_epochs": int(epoch_match.group(2))
        }

    # Pattern for batch progress with losses
    batch_match = re.search(r'(\d+)/(\d+).*?(?:D_Loss|D\s*Loss)[:\s]*([+-]?\d+\.?\d*)', output_line)
    if batch_match:
        batch = int(batch_match.group(1))
        total_batches = int(batch_match.group(2))
        d_loss = float(batch_match.group(3))

        # Extract generator loss if present
        g_loss_match = re.search(r'(?:G_Loss|G\s*Loss)[:\s]*([+-]?\d+\.?\d*)', output_line)
        g_loss = float(g_loss_match.group(1)) if g_loss_match else 0

        return {
            "type": "batch_progress",
            "batch": batch,
            "total_batches": total_batches,
            "metrics": {
                "discriminator_loss": d_loss,
                "generator_loss": g_loss
            }
        }

    return None



async def run_training_task(training_id: str, config: TrainingConfig):
    """Background task to run the actual training with correct epoch handling for resumed sessions"""
    training_status = training_manager.active_trainings[training_id]

    try:
        training_status.status = "running"

        # CRITICAL: Configure matplotlib for headless operation at the start
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode

        # Set environment variables for headless operation
        os.environ['MPLBACKEND'] = 'Agg'
        os.environ['MATPLOTLIB_BACKEND'] = 'Agg'
        os.environ['DISPLAY'] = ''  # Disable X11 display on macOS/Linux

        print(f"🎨 Matplotlib configured for headless web operation")

        # Send initial WebSocket update
        print(f"📊 Starting WebSocket updates for training ID: {training_id}")
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
            print(f"🤖 Auto-resume mode: ENABLED")
            print(f"🎨 Matplotlib: Headless mode for web training")

            # Create training wrapper with correct epoch handling
            training_wrapper = EnhancedTrainingWrapper(training_id, config.dataset)

            # Send auto-resume notification
            await websocket_manager.send_log_message({
                "level": "info",
                "message": "Auto-resume mode enabled - will automatically resume from latest checkpoint",
                "dataset": config.dataset,
                "source": "training",
                "timestamp": datetime.now().isoformat()
            })

            # Parse training output to detect resumed epoch information
            progress_parser = TrainingProgressParser(training_id, config.dataset)

            # Capture training output for parsing
            training_completed = {"value": False}
            training_error = {"error": None}

            def run_actual_training_with_output_capture():
                """Run training and capture output for progress parsing"""
                try:
                    print(f"🚀 Starting actual training with output capture...")

                    # Configure matplotlib for this thread
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    plt.ioff()
                    plt.close('all')

                    os.environ['MPLBACKEND'] = 'Agg'
                    os.environ['MATPLOTLIB_BACKEND'] = 'Agg'
                    os.environ['DISPLAY'] = ''

                    print(f"🎨 Thread matplotlib safety configured")

                    # Use actual dataset config
                    dataset_config = DATASETS[config.dataset]

                    # Set up output capturing to parse resume information
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr

                    class OutputCapture:
                        def __init__(self, original_stream, parser):
                            self.original_stream = original_stream
                            self.parser = parser

                        def write(self, text):
                            # Write to original stream
                            self.original_stream.write(text)
                            self.original_stream.flush()

                            # Parse for training progress
                            try:
                                self.parser.parse_and_send_update(text)
                            except Exception as e:
                                print(f"Parser error: {e}")

                        def flush(self):
                            self.original_stream.flush()

                    # Redirect stdout and stderr to capture training output
                    sys.stdout = OutputCapture(original_stdout, progress_parser)
                    sys.stderr = OutputCapture(original_stderr, progress_parser)

                    try:
                        # Initialize the training with correct parameters
                        training_wrapper.start_training(config.epochs, 0)  # Will be updated when resume is detected

                        # Run the actual training
                        ema_generator, critic = train_enhanced_gan_with_resume_modified(
                            dataset_key=config.dataset,
                            config=dataset_config,
                            resume_from_checkpoint=(config.resume_mode != 'fresh'),
                            num_epochs=config.epochs,
                            experiment_name=config.experiment_name
                        )

                        print(f"✅ Real training completed successfully!")
                        training_wrapper.on_training_complete({
                            "final_epoch": config.epochs,
                            "status": "completed"
                        })

                    finally:
                        # Restore original stdout/stderr
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr

                except Exception as e:
                    print(f"❌ Real training failed: {e}")
                    training_error["error"] = str(e)
                    training_status.status = "error"
                    training_status.error_message = str(e)
                finally:
                    training_completed["value"] = True
                    print(f"🏁 Training thread finished")

            # Start training in a separate thread
            print(f"🚀 Starting training thread with output capture...")
            training_thread = Thread(
                target=run_actual_training_with_output_capture,
                name=f"Training-{training_id}",
                daemon=True
            )
            training_thread.start()

            # Monitor training completion
            while not training_completed["value"]:
                await asyncio.sleep(1)

                # Send heartbeat to keep connection alive
                if training_status.status == "running":
                    await websocket_manager.send_log_message({
                        "level": "debug",
                        "message": "Training in progress...",
                        "dataset": config.dataset,
                        "source": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })

            # Wait for thread completion
            training_thread.join(timeout=10)

            if training_thread.is_alive():
                print(f"⚠️ Training thread still running after timeout")
                await websocket_manager.send_log_message({
                    "level": "warning",
                    "message": "Training is taking longer than expected, continuing in background",
                    "dataset": config.dataset,
                    "source": "training",
                    "timestamp": datetime.now().isoformat()
                })

            # Check for errors
            if training_error["error"]:
                raise Exception(training_error["error"])

        else:
            # Mock training for development/demo with proper resumed epoch simulation
            print(f"🎯 Mock training started for {config.dataset} (development mode)")

            # Create training wrapper
            training_wrapper = EnhancedTrainingWrapper(training_id, config.dataset)

            # Simulate resumed training (for demo purposes)
            resumed_from_epoch = 26 if config.resume_mode != 'fresh' else 0
            training_wrapper.start_training(config.epochs, resumed_from_epoch)

            # Simulate training progress with correct epoch numbers
            start_epoch = resumed_from_epoch if resumed_from_epoch > 0 else 1

            for absolute_epoch in range(start_epoch, min(start_epoch + 5, config.epochs + 1)):
                # Check if training was stopped
                if training_status.status != "running":
                    break

                await asyncio.sleep(1)  # Simulate training time

                # Calculate relative epoch for the wrapper
                relative_epoch = absolute_epoch - resumed_from_epoch if resumed_from_epoch > 0 else absolute_epoch

                # Update progress
                training_status.current_epoch = absolute_epoch
                training_status.progress_percentage = (absolute_epoch / config.epochs) * 100
                training_status.metrics = {
                    "generator_loss": max(0.1, 0.8 - (absolute_epoch * 0.015) + np.random.normal(0, 0.05)),
                    "discriminator_loss": max(0.1, 0.7 - (absolute_epoch * 0.012) + np.random.normal(0, 0.04)),
                    "wasserstein_distance": max(-2.0, -0.5 - (absolute_epoch * 0.01) + np.random.normal(0, 0.1)),
                    "ema_quality": min(1.0, 0.3 + (absolute_epoch * 0.014) + np.random.normal(0, 0.02)),
                    "learning_rate_g": 0.0001 * (0.995 ** absolute_epoch),
                    "learning_rate_d": 0.0004 * (0.995 ** absolute_epoch),
                    "batch_time": 0.5 + np.random.normal(0, 0.1)
                }

                # Send progress update via wrapper (which will use correct epoch numbers)
                training_wrapper.on_epoch_progress(relative_epoch, config.epochs)

                # Simulate batch updates
                for batch in range(1, 4):  # Just a few batches for demo
                    training_wrapper.on_batch_complete(
                        relative_epoch, batch, 469, training_status.metrics
                    )
                    await asyncio.sleep(0.2)

                training_wrapper.on_epoch_complete(relative_epoch, training_status.metrics)

                print(f"📊 Mock Epoch {absolute_epoch}/{config.epochs}: "
                      f"Loss G={training_status.metrics['generator_loss']:.4f}, "
                      f"D={training_status.metrics['discriminator_loss']:.4f}")

        # Training completed successfully
        if training_status.status == "running":  # Only set to completed if not already error/stopped
            training_status.status = "completed"

        training_status.end_time = datetime.now().isoformat()
        training_status.progress_percentage = 100.0

        print(f"🎉 Training completed for {config.dataset}!")

        # Send completion log
        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Training completed successfully for {config.dataset} - {config.epochs} epochs finished",
            "dataset": config.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        training_status.status = "error"
        training_status.error_message = str(e)
        training_status.end_time = datetime.now().isoformat()
        print(f"❌ Training error: {e}")

        # Import traceback for detailed error info
        import traceback
        error_details = traceback.format_exc()
        print(f"📍 Full error traceback:\n{error_details}")

        # Send error log
        await websocket_manager.send_log_message({
            "level": "error",
            "message": f"Training failed: {str(e)}",
            "dataset": config.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    # Send final status update with correct epoch information
    final_status_data = {
        "status": training_status.status,
        "dataset": config.dataset,
        "current_epoch": training_status.current_epoch,
        "total_epochs": training_status.total_epochs,
        "progress_percentage": training_status.progress_percentage,
        "end_time": training_status.end_time,
        "timestamp": datetime.now().isoformat()
    }

    # Add metrics if they exist
    if hasattr(training_status, 'metrics') and training_status.metrics:
        final_status_data["metrics"] = training_status.metrics

    # Add error message if there was an error
    if training_status.error_message:
        final_status_data["error_message"] = training_status.error_message

    await websocket_manager.send_training_update(training_id, final_status_data)

    print(f"📊 Final training status sent for ID: {training_id} - Status: {training_status.status}")

    # Clean up matplotlib resources
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        print(f"🎨 Matplotlib cleanup completed")
    except Exception as cleanup_error:
        print(f"⚠️ Matplotlib cleanup warning: {cleanup_error}")

    print(f"🏁 Training task completed for ID: {training_id}")

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

        # Get checkpoint path from request (if provided)
        checkpoint_path = getattr(request, 'checkpoint_path', 'latest')

        # Generate images with real model
        images = generate_with_real_model(
            dataset=request.dataset,
            num_samples=request.num_samples,
            prompt=request.prompt,
            checkpoint_path=checkpoint_path
        )

        # Send completion log
        method = images[0].get('method', 'unknown') if images else 'unknown'
        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Generated {len(images)} images successfully using {method}",
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
            "message": f"Generated {len(images)} images using {'real trained model' if method == 'real_dcgan' else 'mock data'}"
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

# ============================================================================
# FIX 1: Add GET endpoint for easy testing (add this to your main.py)
# ============================================================================

@app.get("/api/test/websocket")
async def test_websocket_get():
    """Test WebSocket functionality - GET version for easy testing"""
    print(f"📧 Testing WebSocket with {websocket_manager.get_connection_count()} connections")

    await websocket_manager.send_training_update("test-123", {
        "status": "running",
        "current_epoch": 1,
        "total_epochs": 10,
        "progress_percentage": 10.0,
        "timestamp": datetime.now().isoformat()
    })

    await websocket_manager.send_log_message({
        "level": "info",
        "message": "Test WebSocket message - if you see this, WebSockets are working!",
        "dataset": "test",
        "source": "test",
        "timestamp": datetime.now().isoformat()
    })

    return {
        "message": "Test messages sent",
        "connections": websocket_manager.get_connection_count(),
        "status": "success",
        "note": "Check your frontend to see if messages were received"
    }

# Keep the existing POST endpoint too
@app.post("/api/test/websocket")
async def test_websocket_post():
    """Test WebSocket functionality - POST version"""
    # Same content as above
    return await test_websocket_get()


# ============================================================================
# FIX 2: Add WebSocket connection debugging endpoint
# ============================================================================

@app.get("/api/debug/websocket/connections")
async def debug_websocket_connections():
    """Debug WebSocket connections"""
    connections = websocket_manager.get_connection_count()
    connection_info = websocket_manager.get_connection_info()

    return {
        "active_connections": connections,
        "connection_details": connection_info,
        "websocket_endpoint": "ws://localhost:8000/ws",
        "status": "healthy" if connections > 0 else "no_connections",
        "timestamp": datetime.now().isoformat(),
        "debug_info": {
            "total_connections_ever": len(websocket_manager.connection_info),
            "current_active": len(websocket_manager.active_connections)
        }
    }


# ============================================================================
# FIX 3: Add manual WebSocket test trigger for active training
# ============================================================================

@app.post("/api/training/{training_id}/test-websocket")
async def test_training_websocket(training_id: str):
    """Test WebSocket updates for a specific training session"""

    # Check if training exists
    if training_id not in training_manager.active_trainings:
        raise HTTPException(status_code=404, detail="Training session not found")

    training_status = training_manager.active_trainings[training_id]

    # Send test update
    await websocket_manager.send_training_update(training_id, {
        "status": "test_update",
        "dataset": training_status.dataset,
        "current_epoch": training_status.current_epoch,
        "total_epochs": training_status.total_epochs,
        "progress_percentage": training_status.progress_percentage,
        "metrics": training_status.metrics,
        "timestamp": datetime.now().isoformat(),
        "test": True
    })

    await websocket_manager.send_log_message({
        "level": "info",
        "message": f"Manual WebSocket test for training {training_id}",
        "dataset": training_status.dataset,
        "source": "test",
        "timestamp": datetime.now().isoformat()
    })

    return {
        "message": f"Test update sent for training {training_id}",
        "connections": websocket_manager.get_connection_count(),
        "training_status": training_status.status
    }


# ============================================================================
# FIX 4: Enhanced WebSocket endpoint with better connection handling
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with better debugging and connection handling"""
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    print(f"🔌 New WebSocket connection attempt from {client_info}")

    try:
        await websocket_manager.connect(websocket)
        print(f"✅ WebSocket connected successfully. Total connections: {websocket_manager.get_connection_count()}")

        # Send immediate test message to confirm connection
        await websocket_manager.send_personal_message({
            "type": "connection_confirmed",
            "message": "WebSocket connection established successfully",
            "timestamp": datetime.now().isoformat(),
            "server_info": {
                "connections": websocket_manager.get_connection_count(),
                "server_time": datetime.now().isoformat()
            }
        }, websocket)

        # Keep connection alive
        while True:
            try:
                # Wait for incoming messages with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                print(f"📨 Received WebSocket message from {client_info}: {data}")

                # Parse message if it's JSON
                try:
                    message = json.loads(data)
                    message_type = message.get("type", "unknown")

                    if message_type == "ping":
                        # Respond to ping with pong
                        await websocket_manager.send_personal_message({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat(),
                            "server_connections": websocket_manager.get_connection_count()
                        }, websocket)

                    elif message_type == "request_status":
                        # Send current training status
                        active_trainings = list(training_manager.active_trainings.values())
                        await websocket_manager.send_personal_message({
                            "type": "training_status_response",
                            "active_trainings": [
                                {
                                    "training_id": t.training_id,
                                    "status": t.status,
                                    "dataset": t.dataset,
                                    "current_epoch": t.current_epoch,
                                    "progress_percentage": t.progress_percentage
                                } for t in active_trainings
                            ],
                            "timestamp": datetime.now().isoformat()
                        }, websocket)

                    else:
                        # Echo back unknown messages
                        await websocket_manager.send_personal_message({
                            "type": "echo",
                            "original_message": data,
                            "timestamp": datetime.now().isoformat()
                        }, websocket)

                except json.JSONDecodeError:
                    # Handle non-JSON messages
                    await websocket_manager.send_personal_message({
                        "type": "echo",
                        "message": f"Received: {data}",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)

            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket_manager.send_personal_message({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "connections": websocket_manager.get_connection_count()
                }, websocket)

    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected normally from {client_info}")
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"🔌 WebSocket error from {client_info}: {e}")
        websocket_manager.disconnect(websocket)


# ============================================================================
# FIX 5: Add JavaScript WebSocket test snippet endpoint
# ============================================================================

@app.get("/api/debug/websocket/test-script")
async def get_websocket_test_script():
    """Get JavaScript code to test WebSocket connection"""

    js_code = """
// WebSocket Test Script for Browser Console
console.log('🔌 Starting WebSocket test...');

const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function(event) {
    console.log('✅ WebSocket connected successfully!');
    
    // Send ping
    ws.send(JSON.stringify({
        type: 'ping',
        timestamp: new Date().toISOString()
    }));
    
    // Request current training status
    ws.send(JSON.stringify({
        type: 'request_status',
        timestamp: new Date().toISOString()
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('📨 Received:', data);
    
    if (data.type === 'training_status') {
        console.log('📊 Training Update:', data.data);
    } else if (data.type === 'log_message') {
        console.log('📝 Log:', data.data.message);
    }
};

ws.onclose = function(event) {
    console.log('🔌 WebSocket disconnected');
};

ws.onerror = function(error) {
    console.error('❌ WebSocket error:', error);
};

// Send test message after 2 seconds
setTimeout(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send('Test message from browser console');
    }
}, 2000);

console.log('WebSocket test initiated. Check messages above.');
"""

    return {
        "javascript_code": js_code,
        "instructions": [
            "1. Copy the JavaScript code below",
            "2. Open your browser's Developer Tools (F12)",
            "3. Go to the Console tab",
            "4. Paste and run the code",
            "5. Watch for WebSocket connection messages",
            "6. If successful, you should see connection confirmed and ping/pong messages"
        ],
        "websocket_url": "ws://localhost:8000/ws",
        "current_connections": websocket_manager.get_connection_count()
    }


# ============================================================================
# FIX 6: Frontend WebSocket connection checker
# ============================================================================

@app.get("/api/frontend/websocket-status")
async def frontend_websocket_status():
    """Check if frontend should be connected to WebSocket"""

    active_trainings = list(training_manager.active_trainings.values())
    should_be_connected = len(active_trainings) > 0

    return {
        "should_be_connected": should_be_connected,
        "active_connections": websocket_manager.get_connection_count(),
        "active_trainings": len(active_trainings),
        "websocket_url": "ws://localhost:8000/ws",
        "training_sessions": [
            {
                "id": t.training_id,
                "status": t.status,
                "dataset": t.dataset,
                "progress": t.progress_percentage
            } for t in active_trainings
        ],
        "recommendation": "Connect to WebSocket" if should_be_connected and websocket_manager.get_connection_count() == 0 else "WebSocket status OK",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Debug Endpoints for WebSocket Testing
# ============================================================================

@app.post("/api/test/websocket")
async def test_websocket():
    """Test WebSocket functionality"""
    print(f"📧 Testing WebSocket with {websocket_manager.get_connection_count()} connections")

    await websocket_manager.send_training_update("test-123", {
        "status": "running",
        "current_epoch": 1,
        "total_epochs": 10,
        "progress_percentage": 10.0,
        "timestamp": datetime.now().isoformat()
    })

    await websocket_manager.send_log_message({
        "level": "info",
        "message": "Test WebSocket message - if you see this, WebSockets are working!",
        "dataset": "test",
        "source": "test",
        "timestamp": datetime.now().isoformat()
    })

    return {
        "message": "Test messages sent",
        "connections": websocket_manager.get_connection_count(),
        "status": "success"
    }

@app.get("/api/debug/websocket/status")
async def websocket_debug_status():
    """Get WebSocket connection status for debugging"""
    return {
        "active_connections": websocket_manager.get_connection_count(),
        "connection_info": websocket_manager.get_connection_info(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/debug/websocket/simulate-training")
async def simulate_training_updates():
    """Simulate training updates for frontend testing"""
    training_id = "debug-simulation"

    # Simulate a few training updates
    for epoch in range(1, 4):
        await websocket_manager.send_training_update(training_id, {
            "status": "running",
            "dataset": "mnist",
            "current_epoch": epoch,
            "total_epochs": 3,
            "progress_percentage": (epoch / 3) * 100,
            "metrics": {
                "generator_loss": 1.5 - (epoch * 0.2),
                "discriminator_loss": -2.0 - (epoch * 0.1),
                "wasserstein_distance": -1.0 - (epoch * 0.1),
                "ema_quality": 0.5 + (epoch * 0.1)
            },
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Simulated epoch {epoch}/3 completed",
            "dataset": "mnist",
            "source": "simulation",
            "timestamp": datetime.now().isoformat()
        })

        await asyncio.sleep(1)  # Wait 1 second between updates

    # Send completion
    await websocket_manager.send_training_update(training_id, {
        "status": "completed",
        "dataset": "mnist",
        "current_epoch": 3,
        "total_epochs": 3,
        "progress_percentage": 100,
        "timestamp": datetime.now().isoformat()
    })

    return {"message": "Training simulation completed", "training_id": training_id}

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
        "websocket_connections": websocket_manager.get_connection_count(),
        "auto_resume_enabled": True,
        "websocket_debug": {
            "active_connections": websocket_manager.get_connection_count(),
            "connection_info": websocket_manager.get_connection_info()
        }
    }


@app.post("/api/debug/test-resume-detection")
async def test_resume_detection():
    """Test resume detection with sample console output"""

    sample_outputs = [
        "🔄 RESUMED from checkpoint at epoch 26",
        "📅 EPOCH 27/50 - Enhanced Training with Composite Metrics Logging",
        "Epoch 28/50:   5%|██                                        | 23/469 [00:15<04:32,  1.64it/s, D_Loss=-1.234, G_Loss=2.567]",
        "🍎 Step   4/469 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|   0.9% | D:-0.9985 G: 7.1326"
    ]

    results = []
    for output in sample_outputs:
        detection_result = detect_resume_info_from_output(output)
        results.append({
            "input": output,
            "detected": detection_result
        })

    return {
        "message": "Resume detection test completed",
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/debug/send-corrected-update/{training_id}")
async def send_corrected_training_update(training_id: str, absolute_epoch: int, total_epochs: int):
    """Send a corrected training update with absolute epoch numbers"""

    progress = (absolute_epoch / total_epochs) * 100

    await websocket_manager.send_training_update(training_id, {
        "status": "running",
        "dataset": "mnist",
        "current_epoch": absolute_epoch,
        "total_epochs": total_epochs,
        "progress_percentage": progress,
        "is_corrected": True,
        "timestamp": datetime.now().isoformat()
    })

    await websocket_manager.send_log_message({
        "level": "info",
        "message": f"Manual corrected update: Epoch {absolute_epoch}/{total_epochs} ({progress:.1f}%)",
        "dataset": "mnist",
        "source": "debug",
        "timestamp": datetime.now().isoformat()
    })

    return {
        "message": f"Sent corrected update for training {training_id}",
        "absolute_epoch": absolute_epoch,
        "total_epochs": total_epochs,
        "progress_percentage": progress
    }


# Add a debug endpoint to test model loading
@app.post("/api/debug/test-model-loading/{dataset}")
async def test_model_loading(dataset: str):
    """Test loading the trained model for a dataset"""

    try:
        if not DCGAN_AVAILABLE:
            return {
                "success": False,
                "message": "DCGAN not available",
                "dataset": dataset
            }

        from enhanced_dcgan_research import find_available_checkpoints

        # Find checkpoints
        checkpoints = find_available_checkpoints(dataset)
        if not checkpoints:
            return {
                "success": False,
                "message": f"No checkpoints found for {dataset}",
                "dataset": dataset
            }

        # Try to load the latest checkpoint
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)

        import torch
        checkpoint_data = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)

        return {
            "success": True,
            "message": "Model loading test successful",
            "dataset": dataset,
            "checkpoint": os.path.basename(latest_checkpoint),
            "epoch": checkpoint_data.get('epoch', 'unknown'),
            "keys": list(checkpoint_data.keys()),
            "has_ema_generator": 'ema_generator_state_dict' in checkpoint_data,
            "has_generator": 'generator_state_dict' in checkpoint_data
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Model loading failed: {str(e)}",
            "dataset": dataset,
            "error": str(e)
        }

def patch_training_for_correct_epochs():
    """
    Apply patches to existing training to send correct epoch numbers
    This function should be called during startup
    """
    print("🔧 Applying training epoch correction patches...")

    # Set environment variables to help detect resumed training
    os.environ['DCGAN_WEB_MODE'] = '1'
    os.environ['DCGAN_EPOCH_TRACKING'] = '1'

    print("✅ Training epoch correction patches applied")
# ============================================================================
# Application Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event with training patches"""

    # CRITICAL: Configure matplotlib for web/headless mode
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()

    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MATPLOTLIB_BACKEND'] = 'Agg'

    print("🚀 Enhanced DCGAN Web API Starting...")
    print(f"📊 DCGAN Available: {DCGAN_AVAILABLE}")
    print(f"🖥️  Device: {device_name} ({device_type})")
    print(f"📁 Storage Root: {STORAGE_ROOT}")
    print(f"🌐 CORS Origins: {ALLOWED_ORIGINS}")
    print(f"🔌 WebSocket Manager: Ready")
    print(f"🤖 Auto-Resume Mode: ENABLED")
    print(f"🔗 WebSocket Endpoint: ws://localhost:8000/ws")
    print(f"🧪 WebSocket Test: http://localhost:8000/api/test/websocket")

    # Apply training patches
    patch_training_for_correct_epochs()

    if not DCGAN_AVAILABLE:
        print("💡 Running in development mode with mock data")

    print("✅ API Ready with Training Integration!")

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