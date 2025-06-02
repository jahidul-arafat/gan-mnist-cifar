#!/usr/bin/env python3
"""
Enhanced DCGAN Research Framework - Production CLI
==================================================

Enhanced CLI with animated ASCII art banner and professional presentation.
FIXED: Compatible with existing fully_integrated_report_v04.py functions.
"""

import time
import sys
import os
from datetime import datetime

def animated_typewriter_effect(text, delay=0.03):
    """Typewriter effect for text."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)

def print_with_color(text, color_code):
    """Print text with ANSI color codes."""
    print(f"\033[{color_code}m{text}\033[0m")

def show_animated_banner():
    """Display animated ASCII art banner with project information."""

    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')

    # Define colors
    CYAN = "96"
    YELLOW = "93"
    GREEN = "92"
    BLUE = "94"
    MAGENTA = "95"
    WHITE = "97"
    BOLD = "1"

    # ASCII Art Banner
    banner = [
        "╔═══════════════════════════════════════════════════════════════════════════════╗",
        "║                                                                               ║",
        "║  ███████╗███╗   ██╗██╗  ██╗ █████╗ ███╗   ██╗ ██████╗███████╗██████╗        ║",
        "║  ██╔════╝████╗  ██║██║  ██║██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗       ║",
        "║  █████╗  ██╔██╗ ██║███████║███████║██╔██╗ ██║██║     █████╗  ██║  ██║       ║",
        "║  ██╔══╝  ██║╚██╗██║██╔══██║██╔══██║██║╚██╗██║██║     ██╔══╝  ██║  ██║       ║",
        "║  ███████╗██║ ╚████║██║  ██║██║  ██║██║ ╚████║╚██████╗███████╗██████╔╝       ║",
        "║  ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═════╝        ║",
        "║                                                                               ║",
        "║     ██████╗  ██████╗ ██████╗  █████╗ ███╗   ██╗                             ║",
        "║     ██╔══██╗██╔════╝██╔════╝ ██╔══██╗████╗  ██║                             ║",
        "║     ██║  ██║██║     ██║  ███╗███████║██╔██╗ ██║                             ║",
        "║     ██║  ██║██║     ██║   ██║██╔══██║██║╚██╗██║                             ║",
        "║     ██████╔╝╚██████╗╚██████╔╝██║  ██║██║ ╚████║                             ║",
        "║     ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝                             ║",
        "║                                                                               ║",
        "║              ██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗ ║",
        "║              ██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║ ║",
        "║              ██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║ ║",
        "║              ██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║ ║",
        "║              ██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║ ║",
        "║              ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝ ║",
        "║                                                                               ║",
        "╚═══════════════════════════════════════════════════════════════════════════════╝"
    ]

    # Print banner with animation
    print_with_color(banner[0], CYAN)
    for i, line in enumerate(banner[1:], 1):
        if i <= 2:  # Top border
            print_with_color(line, CYAN)
        elif i <= 9:  # ENHANCED text
            print_with_color(line, YELLOW + ";" + BOLD)
        elif i <= 16:  # DCGAN text
            print_with_color(line, GREEN + ";" + BOLD)
        elif i <= 23:  # RESEARCH text
            print_with_color(line, BLUE + ";" + BOLD)
        else:  # Bottom border
            print_with_color(line, CYAN)
        time.sleep(0.05)  # Faster animation

    # Project information
    print()
    print_with_color("┌─────────────────────────── PROJECT INFORMATION ───────────────────────────┐", WHITE)

    info_lines = [
        f"│  🎯 Enhanced Deep Convolutional Generative Adversarial Networks            │",
        f"│  🔬 Academic Research Framework with Complete Image Generation             │",
        f"│  🏗️  Production-Ready Implementation with Advanced Features                │",
        f"│                                                                            │",
        f"│  📅 Version: v0.1.2                                                        │",
        f"│  👨‍💻 Developed by: Enhanced AI Research Team by Jahidul Arafat              │",
        f"│  📧 Contact: jahidapon@gmail.com, Linkedin: https://www.linkedin.com/in/jahidul-arafat-presidential-fellow-phd-candidate-791a7490/│",
        f"│  🌐 Repository: https://github.com/jahidul-arafat/gan-mnist-cifar          │",
        f"│  📄 License: MIT License                                                   │",
        f"│  🕒 Build Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                        │",
    ]

    for line in info_lines:
        print_with_color(line, WHITE)
        time.sleep(0.02)  # Faster animation

    print_with_color("└────────────────────────────────────────────────────────────────────────────┘", WHITE)

    # Features showcase
    print()
    print_with_color("┌──────────────────────── KEY FEATURES & CAPABILITIES ──────────────────────┐", MAGENTA)

    features = [
        "│  ✨ WGAN-GP Loss with Gradient Penalty for Stable Training                 │",
        "│  🧠 Exponential Moving Average (EMA) for Enhanced Quality                  │",
        "│  🏗️  Advanced Generator/Critic Architecture with Spectral Normalization   │",
        "│  💾 Intelligent Checkpoint Management with Auto-Save                       │",
        "│  🔄 Graceful Interrupt Handling (Ctrl+C) with Emergency Recovery          │",
        "│  📊 Real-time Progress Monitoring with Live Visualizations                 │",
        "│  🖼️  Complete Image Generation Integration & Academic Reporting            │",
        "│  🎯 Multi-Device Support: CUDA, Apple Metal MPS, CPU                      │",
        "│  📈 Interactive Generation with Natural Language Prompts                   │",
        "│  📋 Comprehensive Academic Report Generation                                │",
    ]

    for feature in features:
        print_with_color(feature, MAGENTA)
        time.sleep(0.02)  # Faster animation

    print_with_color("└────────────────────────────────────────────────────────────────────────────┘", MAGENTA)

    # Loading animation
    print()
    print_with_color("🚀 Initializing Enhanced DCGAN Research Framework...", YELLOW + ";" + BOLD)

    loading_steps = [
        "🔍 Detecting compute devices",
        "⚙️  Configuring optimizations",
        "📦 Loading neural architectures",
        "🧠 Initializing EMA systems",
        "💾 Setting up checkpoint management",
        "📊 Preparing monitoring systems",
        "🖼️  Configuring image generation",
        "✅ Framework ready for research"
    ]

    for step in loading_steps:
        print(f"   {step}...", end="")
        # Faster loading time
        for i in range(2):
            time.sleep(0.15)
            print(".", end="")
            sys.stdout.flush()
        print(" ✓")

    print()
    print_with_color("=" * 80, GREEN)
    print_with_color("🎉 ENHANCED DCGAN RESEARCH FRAMEWORK SUCCESSFULLY INITIALIZED", GREEN + ";" + BOLD)
    print_with_color("=" * 80, GREEN)
    print()

def show_version_info():
    """Display detailed version information."""
    print_with_color("Enhanced DCGAN Research Framework", "96;1")
    print_with_color("=" * 50, "96")
    print()

    version_info = [
        ("Version", "v0.1.2"),
        ("Release Date", "2024-12-19"),
        ("Build", "Production"),
        ("Python Requirements", ">=3.8"),
        ("PyTorch Requirements", ">=2.0.0"),
        ("License", "MIT"),
        ("Authors", "Enhanced AI Research Team by Jahidul Arafat"),
        ("Repository", "https://github.com/jahidul-arafat/gan-mnist-cifar"),
        ("Documentation", "enhanced-dcgan.readthedocs.io"),
        ("PyPI Package", "enhanced-dcgan-research")
    ]

    for key, value in version_info:
        print(f"  🔹 {key:<20}: {value}")

    print()
    print_with_color("Key Features:", "93;1")
    features = [
        "WGAN-GP with Gradient Penalty",
        "Exponential Moving Average (EMA)",
        "Advanced Checkpoint Management",
        "Multi-Device Support (CUDA/MPS/CPU)",
        "Real-time Progress Monitoring",
        "Interactive Image Generation",
        "Academic Report Generation",
        "Graceful Error Handling"
    ]

    for feature in features:
        print(f"  ✅ {feature}")

    print()
    print_with_color("Device Compatibility:", "92;1")
    devices = [
        "🟢 NVIDIA GPUs (CUDA)",
        "🍎 Apple Silicon (Metal MPS)",
        "💻 Intel/AMD CPUs",
        "☁️  Cloud Computing Platforms"
    ]

    for device in devices:
        print(f"  {device}")

def get_comprehensive_status_with_metrics():
    """Get comprehensive system status including datasets, metrics, and current values."""

    status = {}

    # Basic integration status
    try:
        from .enhanced_dcgan_mnist_cifar_for_apple_mps_checkpoints_graceful_v02 import (
            device, device_name, device_type, recommended_batch_size,
            DATASETS, find_available_checkpoints, memory_manager,
            TQDM_AVAILABLE, TENSORBOARD_AVAILABLE
        )

        status['system'] = {
            'device_available': device is not None,
            'device_name': device_name,
            'device_type': device_type,
            'recommended_batch_size': recommended_batch_size,
            'tqdm_available': TQDM_AVAILABLE,
            'tensorboard_available': TENSORBOARD_AVAILABLE,
            'memory_usage': memory_manager.get_memory_usage() if hasattr(memory_manager, 'get_memory_usage') else 'N/A'
        }

        # Dataset information
        status['datasets'] = {}
        for dataset_key, config in DATASETS.items():
            checkpoints = find_available_checkpoints(dataset_key)

            # Get latest checkpoint info if available
            latest_checkpoint_info = None
            if checkpoints:
                try:
                    import torch
                    latest_checkpoint = checkpoints[0]  # Already sorted by newest
                    checkpoint_data = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
                    latest_checkpoint_info = {
                        'epoch': checkpoint_data.get('epoch', 'Unknown'),
                        'training_stats': checkpoint_data.get('training_stats', {}),
                        'file_size_mb': os.path.getsize(latest_checkpoint) / (1024 * 1024),
                        'file_name': os.path.basename(latest_checkpoint),
                        'file_path': latest_checkpoint,  # ADD THIS LINE - store the full path
                        'timestamp': checkpoint_data.get('timestamp', 'Unknown')
                    }
                except:
                    latest_checkpoint_info = {'error': 'Could not read checkpoint'}

            status['datasets'][dataset_key] = {
                'name': config.name,
                'description': config.description,
                'image_size': f"{config.image_size}x{config.image_size}",
                'channels': config.channels,
                'color_mode': 'Grayscale' if config.channels == 1 else 'RGB',
                'num_classes': config.num_classes,
                'preprocessing': config.preprocessing_info,
                'checkpoints_available': len(checkpoints),
                'latest_checkpoint': latest_checkpoint_info
            }

        # Training metrics definitions
        status['metrics'] = {
            'critic_loss': {
                'name': 'Critic Loss (Discriminator)',
                'description': 'WGAN-GP critic loss measuring real vs fake classification',
                'optimal_range': '< 1.0 (lower is better)',
                'formula': 'E[D(fake)] - E[D(real)] + λ * GP',
                'interpretation': 'Lower values indicate better critic training'
            },
            'generator_loss': {
                'name': 'Generator Loss',
                'description': 'Loss measuring generator ability to fool critic',
                'optimal_range': '< 1.0 (lower is better)',
                'formula': '-E[D(G(z))]',
                'interpretation': 'Lower values indicate better generator training'
            },
            'wasserstein_distance': {
                'name': 'Wasserstein Distance',
                'description': 'Earth mover distance between real and generated distributions',
                'optimal_range': 'Close to 0 (absolute value)',
                'formula': 'E[D(real)] - E[D(fake)]',
                'interpretation': 'Closer to 0 indicates better distribution matching'
            },
            'gradient_penalty': {
                'name': 'Gradient Penalty (GP)',
                'description': 'WGAN-GP penalty term applied to loss',
                'optimal_range': '< 0.1 (lower = better when norms ~1.0)',
                'formula': 'λ * E[(||∇D(x̂)||₂ - 1)²]',
                'interpretation': 'Lower values indicate gradient norms closer to 1.0'
            },
            'gradient_norm': {  # ADD THIS NEW METRIC
                'name': 'Gradient Norm',
                'description': 'L2 norm of critic gradients on interpolated samples',
                'optimal_range': '0.8-1.2 (target: ~1.0)',
                'formula': '||∇D(x̂)||₂',
                'interpretation': 'Should be close to 1.0 for proper Lipschitz constraint'
            },
            'ema_quality': {
                'name': 'EMA Quality Score',
                'description': 'Exponential Moving Average generator quality metric',
                'optimal_range': '0.7-0.9 (higher is better)',
                'formula': 'Custom quality assessment of EMA parameters',
                'interpretation': 'Higher values indicate more stable generation'
            },
            'learning_rate_g': {
                'name': 'Generator Learning Rate',
                'description': 'Current learning rate for generator optimizer',
                'optimal_range': '1e-5 to 1e-3',
                'formula': 'Exponential decay schedule',
                'interpretation': 'Automatically adjusted during training'
            },
            'learning_rate_d': {
                'name': 'Critic Learning Rate',
                'description': 'Current learning rate for critic optimizer',
                'optimal_range': '1e-5 to 1e-3',
                'formula': 'Exponential decay schedule',
                'interpretation': 'Automatically adjusted during training'
            }
        }

        # Enhanced features status
        status['enhanced_features'] = {
            'wgan_gp_loss': {
                'active': True,
                'description': 'Wasserstein GAN with Gradient Penalty',
                'lambda_gp': 10.0,
                'benefits': 'Stable training, no mode collapse'
            },
            'ema_generator': {
                'active': True,
                'description': 'Exponential Moving Average Generator',
                'decay': 0.999,
                'benefits': 'Improved sample quality and stability'
            },
            'spectral_normalization': {
                'active': True,
                'description': 'Spectral normalization on critic layers',
                'target': 'All Conv2d and Linear layers',
                'benefits': 'Enforces Lipschitz constraint'
            },
            'progressive_lr': {
                'active': True,
                'description': 'Progressive learning rate scheduling',
                'decay_factor': 0.995,
                'benefits': 'Better convergence over time'
            },
            'checkpoint_management': {
                'active': True,
                'description': 'Auto-save every 5 epochs with graceful interrupts',
                'frequency': '5 epochs',
                'benefits': 'Training resilience and resumability'
            },
            'device_optimization': {
                'active': True,
                'description': f'Hardware-specific optimizations for {device_type.upper()}',
                'target_device': device_type,
                'benefits': 'Maximum performance utilization'
            },
            'real_time_monitoring': {
                'active': True,
                'description': 'Live progress tracking and visualizations',
                'components': 'Progress bars, live plots, terminal streaming',
                'benefits': 'Training insights and debugging'
            },
            'image_generation': {
                'active': True,
                'description': 'Complete image generation integration',
                'frequency': 'Every 10 epochs + epoch 1',
                'benefits': 'Visual training progress documentation'
            }
        }

        # Class information for each dataset
        from .enhanced_dcgan_mnist_cifar_for_apple_mps_checkpoints_graceful_v02 import get_class_names
        status['class_information'] = {}
        for dataset_key in DATASETS.keys():
            class_names = get_class_names(dataset_key)
            status['class_information'][dataset_key] = {
                'classes': class_names,
                'total_classes': len(class_names),
                'class_mapping': {i: name for i, name in enumerate(class_names)}
            }

    except ImportError as e:
        status['error'] = f"Could not load modules: {e}"

    return status

def get_enhanced_checkpoint_info(checkpoint_path):
    """Get enhanced checkpoint info including file hash"""
    import hashlib
    import os

    try:
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

        # Calculate file hash for verification
        with open(checkpoint_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]  # First 8 chars

        return {
            'file_size_mb': file_size_mb,
            'file_hash': file_hash
        }
    except Exception:
        return {
            'file_size_mb': 0,
            'file_hash': 'unknown'
        }

def display_separate_dataset_metrics_tables(status):
    """Display separate metrics tables for each dataset with FIXED WGAN-GP logic and complete description column"""

    # 3. Training Metrics Reference - SEPARATED BY DATASET
    print_with_color("\n📈 TRAINING METRICS REFERENCE BY DATASET", "94;1")
    print_with_color("─" * 60, "94")

    metrics = status.get('metrics', {})
    datasets = status.get('datasets', {})

    # FIXED mapping for stat keys
    stat_key_map = {
        'critic_loss': 'avg_d_loss',
        'generator_loss': 'avg_g_loss',
        'wasserstein_distance': 'avg_wd',
        'gradient_penalty': 'avg_gp',
        'gradient_norm': 'avg_grad_norm',  # ADDED gradient norm support
        'ema_quality': 'ema_quality',
        'learning_rate_g': 'lr_g',
        'learning_rate_d': 'lr_d'
    }

    # Process each dataset separately
    for dataset_key, dataset_info in datasets.items():
        print_with_color(f"\n🎯 {dataset_info['name']} DATASET METRICS", "94;1")
        print_with_color("─" * 50, "94")

        # Check if this dataset has training data
        has_training_data = (dataset_info.get('latest_checkpoint') and
                             'training_stats' in dataset_info['latest_checkpoint'] and
                             dataset_info['latest_checkpoint']['training_stats'])

        if has_training_data:
            stats = dataset_info['latest_checkpoint']['training_stats']
            epoch = dataset_info['latest_checkpoint'].get('epoch', 'Unknown')

            print(f"📅 Latest Checkpoint: Epoch {epoch}")
            print(f"📁 File: {dataset_info['latest_checkpoint'].get('file_name', 'Unknown')}")

            # FIX: Use the stored file path
            latest_checkpoint_path = dataset_info['latest_checkpoint'].get('file_path')
            if latest_checkpoint_path:
                enhanced_info = get_enhanced_checkpoint_info(latest_checkpoint_path)
                print(f"💾 Size: {enhanced_info['file_size_mb']:.2f} MB (hash: {enhanced_info['file_hash']})")
            else:
                # Fallback
                file_size_mb = dataset_info['latest_checkpoint'].get('file_size_mb', 0)
                print(f"💾 Size: {file_size_mb:.2f} MB")

            print()

            # Create table header with all columns including Description
            print(f"{'Metric':<20} {'Current Value':<15} {'Performance':<25} {'Optimal Range':<15} {'Description'}")
            print("─" * 110)

            # Process each metric for this dataset
            for metric_key, metric_info in metrics.items():
                name = metric_info['name']
                if len(name) > 19:
                    name = name[:16] + "..."

                optimal = metric_info['optimal_range']
                if len(optimal) > 14:
                    optimal = optimal[:11] + "..."

                description = metric_info['description']
                if len(description) > 30:
                    description = description[:27] + "..."

                # Get current value for this specific dataset
                current_value = "No Data"
                performance_status = "❓ Unknown"

                if metric_key in stat_key_map:
                    stat_key = stat_key_map[metric_key]
                    if stat_key in stats:
                        value = stats[stat_key]

                        if isinstance(value, (int, float)):
                            # Format the value appropriately
                            if metric_key.startswith('learning_rate') and value < 0.001:
                                current_value = f"{value:.2e}"
                            elif abs(value) >= 1:
                                current_value = f"{value:.4f}"
                            else:
                                current_value = f"{value:.6f}"

                            # FIXED: Proper logic for WGAN-GP metrics
                            if metric_key == "critic_loss":
                                # FIXED: For WGAN-GP critic loss, more negative = better performance
                                if value <= -3.0:  # Very negative = excellent discrimination
                                    performance_status = "✅ Excellent (very negative)"
                                elif value <= -2.0:  # Very negative = excellent
                                    performance_status = "✅ Excellent (very negative)"
                                elif value <= -1.0:  # Moderately negative = good
                                    performance_status = "✅ Good (negative good)"
                                elif value <= 0:  # Slightly negative = fair
                                    performance_status = "⚠️ Fair (more negative better)"
                                else:  # Positive = bad for WGAN-GP
                                    performance_status = "❌ Poor (should be negative)"

                            elif metric_key == "generator_loss":
                                # FIXED: For WGAN-GP, generator loss = -E[D(G(z))]
                                # Positive values are normal when critic discriminates well
                                if value <= 1.0:
                                    performance_status = "✅ Excellent (low loss)"
                                elif value <= 3.0:
                                    performance_status = "✅ Good (reasonable loss)"
                                elif value <= 6.0:
                                    performance_status = "✅ Normal (healthy adversarial training)"
                                elif value <= 10.0:
                                    performance_status = "⚠️ High (but may be normal for strong critic)"
                                else:
                                    performance_status = "❌ Very high (check training stability)"

                            elif metric_key == "wasserstein_distance":
                                # Wasserstein distance: closer to 0 = better
                                if abs(value) < 0.5:
                                    performance_status = "✅ Excellent (close to 0)"
                                elif abs(value) < 1.0:
                                    performance_status = "✅ Good (close to 0)"
                                elif abs(value) < 2.0:
                                    performance_status = "⚠️ Fair (closer to 0 better)"
                                elif abs(value) < 4.0:
                                    performance_status = "⚠️ High (closer to 0 better)"
                                else:
                                    performance_status = "❌ Poor (far from 0)"

                            elif metric_key == "gradient_penalty":
                                # FIXED: Gradient penalty - lower is better when norms are good
                                if value < 0.01:
                                    performance_status = "✅ Excellent (very low penalty)"
                                elif value < 0.1:
                                    performance_status = "✅ Very good (low penalty)"
                                elif value < 0.5:
                                    performance_status = "✅ Good (reasonable penalty)"
                                elif value < 2.0:
                                    performance_status = "⚠️ Moderate penalty"
                                elif value < 10.0:
                                    performance_status = "⚠️ High penalty (check grad norms)"
                                else:
                                    performance_status = "❌ Very high penalty"

                            elif metric_key == "gradient_norm":
                                # ADDED: Gradient norm evaluation
                                if 0.9 <= value <= 1.1:
                                    performance_status = "✅ Excellent (near 1.0)"
                                elif 0.8 <= value <= 1.2:
                                    performance_status = "✅ Good (within target range)"
                                elif 0.7 <= value <= 1.3:
                                    performance_status = "⚠️ Acceptable (close to range)"
                                elif 0.5 <= value <= 1.5:
                                    performance_status = "⚠️ Fair (outside target)"
                                else:
                                    performance_status = "❌ Poor (far from 1.0)"

                            elif metric_key == "ema_quality":
                                # EMA quality: higher = better
                                if value >= 0.9:
                                    performance_status = "✅ Excellent (higher better)"
                                elif value >= 0.8:
                                    performance_status = "✅ Very good (higher better)"
                                elif value >= 0.7:
                                    performance_status = "✅ Good (higher better)"
                                elif value >= 0.5:
                                    performance_status = "⚠️ Fair (higher better)"
                                else:
                                    performance_status = "❌ Poor (higher better)"

                            elif metric_key.startswith("learning_rate"):
                                # Learning rates: should be in reasonable range
                                if 1e-5 <= value <= 1e-3:
                                    performance_status = "✅ Normal range"
                                elif 1e-6 <= value <= 1e-2:
                                    performance_status = "⚠️ Boundary range"
                                elif value < 1e-6:
                                    performance_status = "❌ Too low"
                                else:
                                    performance_status = "❌ Too high"
                        else:
                            current_value = str(value)
                            performance_status = "ℹ️ Info only"

                print(f"{name:<20} {current_value:<15} {performance_status:<25} {optimal:<15} {description}")

            print("─" * 110)

            # Add dataset-specific summary with FIXED health calculation
            print(f"📊 Dataset Summary:")

            # Calculate overall health for this dataset - FIXED for WGAN-GP
            health_indicators = []
            health_details = []

            if 'avg_d_loss' in stats:
                # FIXED: For WGAN-GP, negative critic loss is good
                d_loss = stats['avg_d_loss']
                is_good = d_loss <= -1.0  # More negative = better
                health_indicators.append(is_good)
                if is_good:
                    if d_loss <= -2.0:
                        health_details.append("✅ Critic loss excellent (very negative)")
                    else:
                        health_details.append("✅ Critic loss good (negative)")
                else:
                    health_details.append("❌ Critic loss needs improvement")

            if 'avg_g_loss' in stats:
                # FIXED: For WGAN-GP, reasonable generator loss range
                g_loss = stats['avg_g_loss']
                is_good = g_loss <= 8.0  # Much more reasonable threshold
                health_indicators.append(is_good)
                if is_good:
                    if g_loss <= 3.0:
                        health_details.append("✅ Generator loss excellent")
                    else:
                        health_details.append("✅ Generator loss normal (healthy adversarial training)")
                else:
                    health_details.append("❌ Generator loss very high")

            if 'avg_wd' in stats:
                wd_good = abs(stats['avg_wd']) < 3.0  # Closer to 0
                health_indicators.append(wd_good)
                if wd_good:
                    if abs(stats['avg_wd']) < 1.0:
                        health_details.append("✅ Wasserstein distance excellent")
                    else:
                        health_details.append("✅ Wasserstein distance reasonable")
                else:
                    health_details.append("❌ Wasserstein distance high")

            if 'avg_gp' in stats:
                gp = stats['avg_gp']
                # FIXED: Updated logic to match the display logic - lower GP is better
                gp_good = gp < 0.5  # Low penalty = good (matches display logic)
                health_indicators.append(gp_good)
                if gp_good:
                    if gp < 0.1:
                        health_details.append("✅ Gradient penalty excellent (very low)")
                    else:
                        health_details.append("✅ Gradient penalty good (low)")
                else:
                    if gp > 10.0:
                        health_details.append("❌ Gradient penalty very high")
                    else:
                        health_details.append("⚠️ Gradient penalty moderate")

            # ADDED: Gradient norm health check
            if 'avg_grad_norm' in stats:
                grad_norm = stats['avg_grad_norm']
                grad_norm_good = 0.8 <= grad_norm <= 1.2
                health_indicators.append(grad_norm_good)
                if grad_norm_good:
                    health_details.append("✅ Gradient norm excellent (near 1.0)")
                else:
                    health_details.append("❌ Gradient norm outside target range")

            if 'ema_quality' in stats:
                ema_good = stats['ema_quality'] >= 0.7
                health_indicators.append(ema_good)
                if ema_good:
                    if stats['ema_quality'] >= 0.9:
                        health_details.append("✅ EMA quality excellent")
                    else:
                        health_details.append("✅ EMA quality good")
                else:
                    health_details.append("❌ EMA quality low")

            if health_indicators:
                health_score = sum(health_indicators) / len(health_indicators) * 100

                if health_score >= 80:
                    health_status = "🟢 Excellent Training Health"
                elif health_score >= 60:
                    health_status = "🟡 Good Training Health"
                elif health_score >= 40:
                    health_status = "🟠 Fair Training Health"
                else:
                    health_status = "🔴 Needs Attention"

                print(f"   🏥 Overall Health: {health_status} ({health_score:.0f}%)")

                # Show health details
                print(f"   📋 Health Details:")
                for detail in health_details:
                    print(f"      {detail}")

            # FIXED: Training recommendations for this dataset
            recommendations = []

            if 'avg_gp' in stats:
                gp = stats['avg_gp']
                # FIXED: Updated recommendations to match new understanding
                if gp > 10.0:
                    recommendations.append("🔧 Decrease gradient penalty lambda (currently too high)")
                elif gp > 2.0:
                    recommendations.append("🔧 Monitor gradient penalty - may be getting high")
                # REMOVED the "increase lambda" recommendation for low GP values
                # Low GP values are GOOD when gradient norms are around 1.0

            if 'avg_wd' in stats and abs(stats['avg_wd']) > 3.0:
                recommendations.append("⏳ Continue training - large Wasserstein distance indicates distributions still learning")

            if 'avg_d_loss' in stats and stats['avg_d_loss'] > 0:
                recommendations.append("🎯 Critic loss is positive - should be negative for WGAN-GP")

            if 'avg_g_loss' in stats and stats['avg_g_loss'] > 8.0:
                recommendations.append("⚡ Very high generator loss - consider adjusting learning rates")

            if 'ema_quality' in stats and stats['ema_quality'] < 0.6:
                recommendations.append("📊 Low EMA quality - monitor training stability")

            # ADDED: Gradient norm recommendations
            if 'avg_grad_norm' in stats:
                grad_norm = stats['avg_grad_norm']
                if grad_norm < 0.7:
                    recommendations.append("📏 Gradient norms too low - consider decreasing GP lambda")
                elif grad_norm > 1.3:
                    recommendations.append("📏 Gradient norms too high - consider increasing GP lambda")

            # Learning rate recommendations
            if 'lr_g' in stats and 'lr_d' in stats:
                lr_g, lr_d = stats['lr_g'], stats['lr_d']
                if lr_g < 1e-6 or lr_d < 1e-6:
                    recommendations.append("📈 Learning rates very low - may need to increase for faster convergence")
                elif lr_g > 1e-2 or lr_d > 1e-2:
                    recommendations.append("📉 Learning rates high - may need to decrease for stability")

            if recommendations:
                print(f"   💡 Recommendations:")
                for rec in recommendations:
                    print(f"      • {rec}")
            else:
                print(f"   ✅ Training progressing well - no immediate actions needed")

        else:
            # No training data available for this dataset
            print(f"❌ No training data available for {dataset_info['name']}")
            print(f"💡 To see metrics for this dataset:")
            print(f"   1. Start training: enhanced-dcgan --dataset {dataset_key} --epochs 25")
            print(f"   2. Or resume existing: enhanced-dcgan --dataset {dataset_key} --resume latest")
            print(f"   3. Training will generate checkpoints with metrics")

        print()  # Add spacing between datasets

def display_comprehensive_status():
    """Display comprehensive status with detailed metrics and dataset information."""

    status = get_comprehensive_status_with_metrics()

    if 'error' in status:
        print_with_color(f"❌ Error getting status: {status['error']}", "91")
        return

    # Header
    print_with_color("\n🔍 ENHANCED DCGAN RESEARCH FRAMEWORK - COMPREHENSIVE STATUS", "96;1")
    print_with_color("=" * 80, "96")

    # 1. System Information
    print_with_color("\n📱 SYSTEM CONFIGURATION", "93;1")
    print_with_color("─" * 40, "93")

    system = status.get('system', {})
    system_table = [
        ["Device Type", system.get('device_type', 'Unknown').upper()],
        ["Device Name", system.get('device_name', 'Unknown')],
        ["Memory Usage", system.get('memory_usage', 'N/A')],
        ["Recommended Batch Size", str(system.get('recommended_batch_size', 'Unknown'))],
        ["Progress Bars (tqdm)", "✅ Available" if system.get('tqdm_available') else "❌ Not Available"],
        ["TensorBoard", "✅ Available" if system.get('tensorboard_available') else "❌ Not Available"]
    ]

    for row in system_table:
        print(f"  {row[0]:<25}: {row[1]}")

    # 2. Dataset Information
    print_with_color("\n📊 DATASET INFORMATION & AVAILABILITY", "92;1")
    print_with_color("─" * 50, "92")

    datasets = status.get('datasets', {})
    for dataset_key, dataset_info in datasets.items():
        print_with_color(f"\n🎯 {dataset_info['name']} Dataset", "92")

        dataset_table = [
            ["Description", dataset_info['description'][:60] + "..." if len(dataset_info['description']) > 60 else dataset_info['description']],
            ["Image Size", dataset_info['image_size']],
            ["Color Mode", f"{dataset_info['color_mode']} ({dataset_info['channels']} channel{'s' if dataset_info['channels'] > 1 else ''})"],
            ["Number of Classes", str(dataset_info['num_classes'])],
            ["Preprocessing", dataset_info['preprocessing'][:50] + "..." if len(dataset_info['preprocessing']) > 50 else dataset_info['preprocessing']],
            ["Available Checkpoints", str(dataset_info['checkpoints_available'])]
        ]

        for row in dataset_table:
            print(f"    {row[0]:<20}: {row[1]}")

        # Latest checkpoint info
        if dataset_info['latest_checkpoint'] and 'error' not in dataset_info['latest_checkpoint']:
            latest = dataset_info['latest_checkpoint']
            print(f"    {'Latest Checkpoint':<20}: {latest.get('file_name', 'Unknown')}")
            print(f"    {'Checkpoint Epoch':<20}: {latest.get('epoch', 'Unknown')}")
            print(f"    {'File Size':<20}: {latest.get('file_size_mb', 0):.1f} MB")

            # Display latest training metrics if available
            if 'training_stats' in latest and latest['training_stats']:
                stats = latest['training_stats']
                print(f"    {'Latest Metrics':<20}:")
                if 'avg_d_loss' in stats:
                    print(f"      {'• Critic Loss':<18}: {stats['avg_d_loss']:.6f}")
                if 'avg_g_loss' in stats:
                    print(f"      {'• Generator Loss':<18}: {stats['avg_g_loss']:.6f}")
                if 'avg_wd' in stats:
                    print(f"      {'• Wasserstein Dist':<18}: {stats['avg_wd']:.6f}")
                if 'ema_quality' in stats:
                    print(f"      {'• EMA Quality':<18}: {stats['ema_quality']:.4f}")
                # FIXED: Now these should show actual values
                if 'lr_g' in stats:
                    print(f"      {'• Generator LR':<18}: {stats['lr_g']:.2e}")
                if 'lr_d' in stats:
                    print(f"      {'• Critic LR':<18}: {stats['lr_d']:.2e}")

        # Class information
        class_info = status.get('class_information', {}).get(dataset_key, {})
        if 'classes' in class_info:
            classes_str = ", ".join(class_info['classes'][:8])  # Show first 8 classes
            if len(class_info['classes']) > 8:
                classes_str += f", ... ({class_info['total_classes']} total)"
            print(f"    {'Classes':<20}: {classes_str}")

    # 3. REPLACE THIS SECTION WITH SEPARATE DATASET TABLES
    display_separate_dataset_metrics_tables(status)

    # 4. Enhanced Features Status (keep existing...)
    print_with_color("\n⚡ ENHANCED FEATURES STATUS", "95;1")
    print_with_color("─" * 40, "95")

    features = status.get('enhanced_features', {})

    print(f"{'Feature':<25} {'Status':<10} {'Configuration':<20} {'Benefits'}")
    print_with_color("─" * 80, "95")

    for feature_key, feature_info in features.items():
        name = feature_info['description'].split(' - ')[0]  # Get main name
        if len(name) > 24:
            name = name[:21] + "..."

        status_icon = "✅ Active" if feature_info['active'] else "❌ Inactive"

        # Get configuration info
        config_info = ""
        if 'lambda_gp' in feature_info:
            config_info = f"λ={feature_info['lambda_gp']}"
        elif 'decay' in feature_info:
            config_info = f"decay={feature_info['decay']}"
        elif 'decay_factor' in feature_info:
            config_info = f"factor={feature_info['decay_factor']}"
        elif 'frequency' in feature_info:
            config_info = feature_info['frequency']
        elif 'target_device' in feature_info:
            config_info = feature_info['target_device'].upper()

        if len(config_info) > 19:
            config_info = config_info[:16] + "..."

        benefits = feature_info['benefits']
        if len(benefits) > 35:
            benefits = benefits[:32] + "..."

        print(f"{name:<25} {status_icon:<10} {config_info:<20} {benefits}")

    # 5. Training Recommendations
    print_with_color("\n💡 TRAINING RECOMMENDATIONS", "93;1")
    print_with_color("─" * 35, "93")

    recommendations = []

    # Check if any checkpoints exist
    total_checkpoints = sum(info['checkpoints_available'] for info in datasets.values())
    if total_checkpoints == 0:
        recommendations.append("🆕 No checkpoints found - start with fresh training")
    else:
        recommendations.append(f"🔄 {total_checkpoints} checkpoints available - consider resuming training")

    # Device-specific recommendations
    device_type = system.get('device_type', '').lower()
    if device_type == 'mps':
        recommendations.append("🍎 Apple Metal GPU detected - optimized for M1/M2/M3 chips")
        recommendations.append(f"📦 Recommended batch size: {system.get('recommended_batch_size', 64)}")
    elif device_type == 'cuda':
        recommendations.append("🟢 NVIDIA GPU detected - high performance training available")
        recommendations.append(f"📦 Recommended batch size: {system.get('recommended_batch_size', 128)}")
    else:
        recommendations.append("💻 CPU training - consider smaller batch sizes for stability")

    # Feature recommendations
    if not system.get('tensorboard_available'):
        recommendations.append("📊 Install tensorboard for training visualization: pip install tensorboard")

    for rec in recommendations:
        print(f"  {rec}")

    # 6. Quick Start Commands
    print_with_color("\n🚀 QUICK START COMMANDS", "96;1")
    print_with_color("─" * 30, "96")

    commands = [
        ("Fresh MNIST Training", "enhanced-dcgan --dataset mnist --epochs 50 --resume fresh"),
        ("Resume Latest Training", "enhanced-dcgan --resume latest"),
        ("Interactive Mode", "enhanced-dcgan --interactive"),
        ("Quick Demo", "enhanced-dcgan --demo"),
        ("Generate Report", "enhanced-dcgan --dataset mnist --epochs 25")
    ]

    for desc, cmd in commands:
        print(f"  {desc:<25}: {cmd}")

    print_with_color("\n" + "=" * 80, "96")
    print_with_color("=" * 80, "96")

    # 7. Training Health Assessment - FIXED
    print_with_color("\n🏥 TRAINING HEALTH ASSESSMENT", "91;1")
    print_with_color("─" * 35, "91")

    for dataset_key, dataset_info in datasets.items():
        if (dataset_info.get('latest_checkpoint') and
                'training_stats' in dataset_info['latest_checkpoint']):

            print_with_color(f"\n🎯 {dataset_info['name']} Health Analysis", "91")

            stats = dataset_info['latest_checkpoint']['training_stats']
            epoch = dataset_info['latest_checkpoint'].get('epoch', 'Unknown')

            # Analyze training health - FIXED LOGIC
            health_issues = []
            health_good = []

            # FIXED: Check each metric with correct WGAN-GP logic
            if 'avg_d_loss' in stats:
                d_loss = stats['avg_d_loss']  # FIXED: Don't take absolute value
                # FIXED: For WGAN-GP, more negative critic loss = better
                if d_loss > 0:
                    health_issues.append(f"Positive critic loss ({d_loss:.2f}) - should be negative for WGAN-GP")
                elif d_loss > -1.0:
                    health_issues.append(f"Critic loss too close to zero ({d_loss:.2f}) - weak discrimination")
                elif d_loss <= -2.0:
                    health_good.append(f"Critic loss excellent ({d_loss:.2f}) - strong discrimination")
                else:  # -2.0 < d_loss <= -1.0
                    health_good.append(f"Critic loss good ({d_loss:.2f})")

            if 'avg_g_loss' in stats:
                g_loss = stats['avg_g_loss']  # FIXED: Don't take absolute value
                # FIXED: Updated thresholds for WGAN-GP generator loss
                if g_loss > 10.0:
                    health_issues.append(f"Very high generator loss ({g_loss:.2f}) - check training stability")
                elif g_loss > 6.0:
                    health_issues.append(f"High generator loss ({g_loss:.2f}) - monitor training progress")
                elif g_loss <= 3.0:
                    health_good.append(f"Generator loss excellent ({g_loss:.2f})")
                else:  # 3.0 < g_loss <= 6.0
                    health_good.append(f"Generator loss normal ({g_loss:.2f}) - healthy adversarial training")

            if 'avg_wd' in stats:
                wd = abs(stats['avg_wd'])
                if wd > 2.0:
                    health_issues.append(f"High Wasserstein distance ({wd:.2f}) - distributions far apart")
                elif wd < 0.5:
                    health_good.append(f"Excellent distribution matching (WD: {wd:.2f})")

            # FIXED: Gradient penalty logic
            if 'avg_gp' in stats:
                gp = stats['avg_gp']
                # FIXED: Low GP values are GOOD when gradient norms are ~1.0
                if gp > 10.0:
                    health_issues.append(f"Very high gradient penalty ({gp:.2f}) - gradient norms poor")
                elif gp > 2.0:
                    health_issues.append(f"High gradient penalty ({gp:.2f}) - check gradient norms")
                elif gp < 0.1:
                    health_good.append(f"Gradient penalty excellent ({gp:.3f}) - gradient norms near 1.0")
                else:  # 0.1 <= gp <= 2.0
                    health_good.append(f"Gradient penalty good ({gp:.3f})")

            if 'ema_quality' in stats:
                ema = stats['ema_quality']
                if ema >= 0.8:
                    health_good.append(f"Excellent EMA quality ({ema:.3f})")
                elif ema < 0.6:
                    health_issues.append(f"Low EMA quality ({ema:.3f}) - may need more stable training")

            # Display health status
            if health_good:
                print("    ✅ Positive Indicators:")
                for item in health_good:
                    print(f"       • {item}")

            if health_issues:
                print("    ⚠️  Areas for Improvement:")
                for item in health_issues:
                    print(f"       • {item}")

            # Overall health score
            total_metrics = len(health_good) + len(health_issues)
            if total_metrics > 0:
                health_score = len(health_good) / total_metrics * 100

                if health_score >= 80:
                    health_status = "🟢 Excellent"
                elif health_score >= 60:
                    health_status = "🟡 Good"
                elif health_score >= 40:
                    health_status = "🟠 Fair"
                else:
                    health_status = "🔴 Needs Attention"

                print(f"    📊 Overall Health: {health_status} ({health_score:.0f}%)")

            # FIXED: Specific recommendations based on correct WGAN-GP understanding
            print("    💊 Recommended Actions:")
            recommendations = []

            # FIXED: Only recommend increasing GP lambda if it's actually too high
            if 'avg_gp' in stats:
                gp = stats['avg_gp']
                if gp > 10.0:
                    recommendations.append("Decrease GP lambda - gradient penalty too high")
                elif gp > 2.0:
                    recommendations.append("Monitor gradient penalty - may be getting high")
                # REMOVED incorrect "increase lambda" recommendation for low GP

            if 'avg_wd' in stats and abs(stats['avg_wd']) > 2.0:
                recommendations.append("Continue training - distributions still learning to match")
                recommendations.append("Consider reducing learning rates by 0.5x")

            if 'avg_d_loss' in stats and stats['avg_d_loss'] > 0:
                recommendations.append("Critic loss is positive - should be negative for WGAN-GP")

            if 'avg_g_loss' in stats and stats['avg_g_loss'] > 8.0:
                recommendations.append("Very high generator loss - consider adjusting learning rates")

            if len(health_issues) > len(health_good):
                recommendations.append("Monitor training more closely")
                recommendations.append("Consider checkpoint rollback if metrics worsen")

            if epoch and isinstance(epoch, int) and epoch < 50:
                recommendations.append(f"Continue training - only at epoch {epoch}, needs more time")

            if recommendations:
                for rec in recommendations:
                    print(f"       • {rec}")
            else:
                print(f"       • Training progressing excellently - no actions needed")

    # 8. Executive Summary
    display_executive_summary(datasets, system, features)


def generate_training_timeline_summary(datasets):
    """Generate a detailed training timeline summary."""

    print_with_color("\n📅 TRAINING TIMELINE SUMMARY", "95;1")
    print_with_color("─" * 35, "95")

    for dataset_key, dataset_info in datasets.items():
        dataset_name = dataset_info['name']
        checkpoints_count = dataset_info['checkpoints_available']

        print(f"\n🎯 {dataset_name}:")

        if dataset_info.get('latest_checkpoint'):
            latest = dataset_info['latest_checkpoint']
            epoch = latest.get('epoch', 'Unknown')
            timestamp = latest.get('timestamp', 'Unknown')
            file_size = latest.get('file_size_mb', 0)

            print(f"    📊 Current Progress: Epoch {epoch}")
            print(f"    💾 Available Checkpoints: {checkpoints_count}")
            print(f"    📁 Latest Checkpoint: {file_size:.2f} MB")
            print(f"    🕒 Last Updated: {timestamp}")

            # Training health indicator
            if 'training_stats' in latest and latest['training_stats']:
                stats = latest['training_stats']
                ema_quality = stats.get('ema_quality', 0)
                avg_wd = abs(stats.get('avg_wd', 0)) if stats.get('avg_wd') is not None else float('inf')

                if ema_quality >= 0.8 and avg_wd < 1.0:
                    health = "🟢 Excellent"
                elif ema_quality >= 0.6:
                    health = "🟡 Good"
                else:
                    health = "🔴 Needs Attention"

                print(f"    🏥 Training Health: {health}")

                # Estimated completion (rough calculation)
                if isinstance(epoch, int) and epoch > 0:
                    if avg_wd > 1.0:  # Still learning
                        estimated_epochs_needed = max(50 - epoch, 10)
                        print(f"    ⏱️  Estimated Training: ~{estimated_epochs_needed} more epochs recommended")
                    else:
                        print(f"    ✅ Training Status: Well converged")
        else:
            print(f"    📊 Status: No training data available")
            print(f"    💡 Recommendation: Start fresh training")

def generate_dynamic_training_progress(datasets):
    """Generate dynamic training progress summary for executive summary."""
    progress_parts = []

    for dataset_key, dataset_info in datasets.items():
        dataset_name = dataset_info['name']

        if dataset_info.get('latest_checkpoint'):
            epoch = dataset_info['latest_checkpoint'].get('epoch', 'Unknown')

            # Add health indicator based on training stats
            if 'training_stats' in dataset_info['latest_checkpoint']:
                stats = dataset_info['latest_checkpoint']['training_stats']
                ema_quality = stats.get('ema_quality', 0)

                if ema_quality >= 0.8:
                    status = "✅"
                elif ema_quality >= 0.6:
                    status = "🟡"
                else:
                    status = "❌"

                progress_parts.append(f"{dataset_name} (Epoch {epoch}) {status}")
            else:
                progress_parts.append(f"{dataset_name} (Epoch {epoch})")
        else:
            checkpoints_count = dataset_info['checkpoints_available']
            if checkpoints_count > 0:
                progress_parts.append(f"{dataset_name}: {checkpoints_count} checkpoints")
            else:
                progress_parts.append(f"{dataset_name}: No data")

    return ", ".join(progress_parts) if progress_parts else "No training data available"

# COMPLETE EXECUTIVE SUMMARY SECTION
# Add this at the end of your display_comprehensive_status() function:

def display_executive_summary(datasets, system, features):
    """Display comprehensive executive summary with dynamic data."""

    # Generate training timeline first
    generate_training_timeline_summary(datasets)

    # Then display executive summary
    print_with_color("\n📋 EXECUTIVE SUMMARY", "96;1")
    print_with_color("─" * 25, "96")

    total_checkpoints = sum(info['checkpoints_available'] for info in datasets.values())
    datasets_with_metrics = len([d for d in datasets.values()
                                 if d.get('latest_checkpoint') and 'training_stats' in d['latest_checkpoint']])

    # Dynamic training progress
    training_progress = generate_dynamic_training_progress(datasets)

    # Calculate overall system health
    active_features = len([f for f in features.values() if f.get('active', False)])
    total_features = len(features)
    system_health_percent = (active_features / total_features * 100) if total_features > 0 else 0

    if system_health_percent == 100:
        system_health = "🟢 Fully Operational"
    elif system_health_percent >= 80:
        system_health = "🟡 Mostly Operational"
    else:
        system_health = "🔴 Partial Operation"

    # Determine overall training status
    total_training_epochs = 0
    datasets_in_training = 0

    for dataset_info in datasets.values():
        if dataset_info.get('latest_checkpoint') and isinstance(dataset_info['latest_checkpoint'].get('epoch'), int):
            total_training_epochs += dataset_info['latest_checkpoint']['epoch']
            datasets_in_training += 1

    if datasets_in_training > 0:
        avg_epoch = total_training_epochs / datasets_in_training
        if avg_epoch >= 30:
            training_maturity = "🎓 Advanced"
        elif avg_epoch >= 15:
            training_maturity = "📈 Intermediate"
        elif avg_epoch >= 5:
            training_maturity = "🌱 Early Stage"
        else:
            training_maturity = "🆕 Just Started"
    else:
        training_maturity = "⏸️ Not Started"

    summary_points = [
        f"📊 {len(datasets)} datasets configured, {datasets_with_metrics} with active training data",
        f"💾 {total_checkpoints} total checkpoints available across all datasets",
        f"🖥️  Running on {system.get('device_type', 'Unknown').upper()} with optimal configuration",
        f"⚡ {active_features}/{total_features} enhanced features active - {system_health}",
        f"📈 Training progress: {training_progress}",
        f"🎯 Training maturity: {training_maturity}",
        f"🔬 Framework status: Ready for continued research and development"
    ]

    for point in summary_points:
        print(f"  {point}")

    # Overall readiness assessment
    readiness_factors = []

    if system.get('device_available', False):
        readiness_factors.append("✅ Hardware")
    else:
        readiness_factors.append("❌ Hardware")

    if total_checkpoints > 0:
        readiness_factors.append("✅ Checkpoints")
    else:
        readiness_factors.append("⚠️ No Checkpoints")

    if system_health_percent == 100:
        readiness_factors.append("✅ Features")
    else:
        readiness_factors.append("⚠️ Features")

    if datasets_with_metrics > 0:
        readiness_factors.append("✅ Training Data")
    else:
        readiness_factors.append("❌ Training Data")

    readiness_score = len([f for f in readiness_factors if f.startswith("✅")]) / len(readiness_factors) * 100

    if readiness_score == 100:
        overall_status = "🚀 FULLY OPERATIONAL"
        status_color = "92;1"  # Green bold
    elif readiness_score >= 75:
        overall_status = "⚡ MOSTLY READY"
        status_color = "93;1"  # Yellow bold
    else:
        overall_status = "🔧 NEEDS SETUP"
        status_color = "91;1"  # Red bold

    print(f"\n📊 Readiness Assessment: {' | '.join(readiness_factors)}")
    print(f"🎯 Readiness Score: {readiness_score:.0f}%")

    print_with_color(f"\n🔬 Research Framework Status: {overall_status}", status_color)

    # Quick action recommendations based on status
    if readiness_score < 100:
        print_with_color("\n💡 RECOMMENDED ACTIONS:", "93;1")

        if not system.get('device_available', False):
            print("   🔧 Check device configuration and drivers")

        if total_checkpoints == 0:
            print("   🆕 Start initial training to create checkpoints")

        if datasets_with_metrics == 0:
            print("   📊 Run training to generate performance metrics")

        if system_health_percent < 100:
            inactive_features = [name for name, info in features.items() if not info.get('active', False)]
            if inactive_features:
                print(f"   ⚡ Activate remaining features: {', '.join(inactive_features[:3])}")

# USAGE: Replace your existing executive summary section with:
# display_executive_summary(datasets, system, features)

def main():
    """Enhanced main entry point with animated banner - FIXED for compatibility."""
    import argparse

    # Create parser - UPDATED to match your existing description
    parser = argparse.ArgumentParser(
        description='FIXED Fully Integrated Enhanced DCGAN Academic Research with Complete Image Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  enhanced-dcgan                           # Interactive mode with banner
  enhanced-dcgan --dataset mnist --epochs 50   # Train MNIST for 50 epochs  
  enhanced-dcgan --demo                    # Quick demo
  enhanced-dcgan --version                 # Show version information
  enhanced-dcgan --no-banner               # Skip animated banner

For more information, visit: https://github.com/jahidul-arafat/gan-mnist-cifar
        """
    )

    # FIXED: Add all your existing arguments
    parser.add_argument('--version', action='store_true',
                        help='Show version information and exit')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'],
                        help='Dataset to use for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--resume', choices=['interactive', 'latest', 'fresh'],
                        default='interactive', help='Resume mode (default: interactive)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--demo', action='store_true',
                        help='Run quick demo of image generation capabilities')
    parser.add_argument('--test', action='store_true',
                        help='Run comprehensive integration test')
    parser.add_argument('--status', action='store_true',
                        help='Show integration status including image generation')

    # NEW: Additional options for enhanced CLI
    parser.add_argument('--no-banner', action='store_true',
                        help='Skip animated banner (for faster startup)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode - minimal output')

    args = parser.parse_args()

    # Handle version flag first
    if args.version:
        show_version_info()
        return

    # Show animated banner unless skipped or in quiet mode
    if not args.no_banner and not args.quiet:
        show_animated_banner()

    # FIXED: Import after banner to avoid premature device output during banner
    try:
        # Import your existing functions - UPDATED for correct import path
        from .enhanced_dcgan_mnist_cifar_for_apple_mps_checkpoints_graceful_v02 import (
            DATASETS, get_dataset_choice
        )
        from .fully_integrated_report_v04 import (
            run_fixed_fully_integrated_academic_study,
            quick_demo_with_images,
            run_integration_test_with_images,
            get_integration_status_with_images
        )

    except ImportError:
        try:
            # Fallback for standalone execution
            from enhanced_dcgan_mnist_cifar_for_apple_mps_checkpoints_graceful_v02 import (
                DATASETS, get_dataset_choice
            )
            from fully_integrated_report_v04 import (
                run_fixed_fully_integrated_academic_study,
                quick_demo_with_images,
                run_integration_test_with_images,
                get_integration_status_with_images
            )
        except ImportError as e:
            print_with_color(f"❌ Import Error: {e}", "91;1")
            if not args.quiet:
                print("💡 Make sure all dependencies are installed")
            return

    # FIXED: Handle special commands exactly like your original
    if args.demo:
        try:
            quick_demo_with_images()
        except NameError:
            print("Demo function not available")
        sys.exit(0)  # FIXED: Add sys.exit(0) like original

    if args.test:
        try:
            run_integration_test_with_images()
        except NameError:
            print("Test function not available")
        sys.exit(0)  # FIXED: Add sys.exit(0) like original

    if args.status:
        try:
            display_comprehensive_status()
        except Exception as e:
            print_with_color(f"❌ Status check failed: {e}", "91")
            print("💡 Some information may not be available")
        sys.exit(0)

    # FIXED: Main execution modes - match your original logic exactly
    if args.interactive or args.dataset is None:
        # Interactive mode
        if not args.quiet:
            # FIXED: Use your exact original text
            print("\n🎓 Welcome to FIXED Fully Integrated Enhanced DCGAN Academic Research!")
            print("🖼️ Complete image generation integration FIXED")
            print("🔗 Complete integration with existing enhanced DCGAN pipeline")
            print("📊 All generated images will be captured and included in reports")
            print("\nChoose your research configuration:")

        if args.dataset is None:
            dataset_choice = get_dataset_choice()
        else:
            dataset_choice = args.dataset

        if not args.quiet:
            print(f"\n✅ Selected dataset: {DATASETS[dataset_choice].name}")

        # FIXED: Ask about resume mode - exact original logic
        if not args.resume or args.resume == 'interactive':
            if not args.quiet:
                print("\n💾 Checkpoint Resume Options:")
                print("1. interactive - Choose from available checkpoints")
                print("2. latest - Auto-resume from latest checkpoint")
                print("3. fresh - Start fresh training (ignore checkpoints)")

            while True:
                resume_choice = input("\nResume mode (interactive/latest/fresh): ").strip().lower()
                if resume_choice in ['interactive', 'latest', 'fresh']:
                    resume_mode = resume_choice
                    break
                print("Please enter 'interactive', 'latest', or 'fresh'")
        else:
            resume_mode = args.resume

        # FIXED: Ask about epochs - exact original logic
        while True:
            try:
                epochs_input = input(f"\nNumber of epochs (default {args.epochs}): ").strip()
                epochs = int(epochs_input) if epochs_input else args.epochs
                if epochs > 0:
                    break
                print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")

        if not args.quiet:
            # FIXED: Use your exact original text
            print(f"\n🚀 Starting FIXED fully integrated academic research study with complete image generation...")
            print(f"   Dataset: {dataset_choice}")
            print(f"   Epochs: {epochs}")
            print(f"   Resume Mode: {resume_mode}")
            print(f"   Integration: Complete (100% feature utilization + image generation)")
            print(f"   🖼️ Image Generation: FIXED - All images will be captured and documented")

        # FIXED: Run the study - exact original call
        reporter, final_report = run_fixed_fully_integrated_academic_study(
            dataset_choice=dataset_choice,
            num_epochs=epochs,
            resume_mode=resume_mode
        )

        if final_report:
            # FIXED: Use your exact original success text
            print(f"\n🎉 STUDY COMPLETED SUCCESSFULLY!")
            print(f"📄 View complete report: {final_report}")
            print(f"🖼️ View generated images: {reporter.report_dir}/generated_samples/")
            print(f"📊 All training data and images documented for academic use")

    else:
        # FIXED: Command line mode - exact original logic
        if not args.quiet:
            print(f"🎓 Running FIXED Fully Integrated Enhanced DCGAN Academic Study")
            print(f"   Dataset: {args.dataset}")
            print(f"   Epochs: {args.epochs}")
            print(f"   Resume Mode: {args.resume}")
            print(f"   Integration: Complete + Image Generation")
            print(f"   🖼️ Image Generation: FIXED")

        reporter, final_report = run_fixed_fully_integrated_academic_study(
            dataset_choice=args.dataset,
            num_epochs=args.epochs,
            resume_mode=args.resume
        )

        if final_report:
            # FIXED: Use your exact original success text
            print(f"\n✅ Academic study completed with complete image documentation!")
            print(f"📄 Report: {final_report}")
            print(f"🖼️ Images: {reporter.report_dir}/generated_samples/")

if __name__ == "__main__":
    main()