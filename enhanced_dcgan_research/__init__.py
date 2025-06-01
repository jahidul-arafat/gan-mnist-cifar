"""
Enhanced DCGAN Research Framework
=================================

A comprehensive framework for Enhanced Deep Convolutional Generative Adversarial Networks
with academic reporting, advanced checkpointing, and production-ready features.
"""

__version__ = "0.1.1"
__author__ = "Jahidul Arafat"
__email__ = "jahidapon@gmail.com"

# Import main functions from your existing modules
try:
    from .enhanced_dcgan_mnist_cifar_for_apple_mps_checkpoints_graceful_v02 import (
        # Core models
        EnhancedConditionalGenerator,
        EnhancedConditionalCritic,
        EMAGenerator,
        WassersteinGPLoss,

        # Training function
        train_enhanced_gan_with_resume_modified,

        # Dataset utilities
        DATASETS,
        get_dataset_choice,
        get_transforms,
        get_dataset,
        get_class_names,

        # Checkpoint management
        find_available_checkpoints,
        get_checkpoint_choice,
        load_checkpoint_and_resume,
        save_checkpoint_enhanced,

        # Device utilities
        device,
        device_name,
        device_type,
        detect_and_setup_device,
        setup_device_optimizations,
        recommended_batch_size,

        # Progress tracking
        ProgressTracker,
        LiveTerminalMonitor,
        EnhancedLivePlotter,
    )

    from .fully_integrated_report_v04 import (
        # Academic reporting
        FixedFullyIntegratedAcademicReporter,
        InteractiveDigitGenerator,
        run_fixed_fully_integrated_academic_study,
        save_academic_generated_images,
    )

    _imports_successful = True

except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    _imports_successful = False

# Main convenience functions
def train_enhanced_gan(dataset_key, num_epochs=50, resume_mode='interactive'):
    """
    Convenience function for training Enhanced DCGAN.

    Args:
        dataset_key (str): 'mnist' or 'cifar10'
        num_epochs (int): Number of training epochs
        resume_mode (str): 'interactive', 'latest', or 'fresh'

    Returns:
        tuple: (ema_generator, critic)
    """
    if not _imports_successful:
        raise ImportError("Required modules not available")

    config = DATASETS[dataset_key]
    return train_enhanced_gan_with_resume_modified(
        dataset_key=dataset_key,
        config=config,
        resume_from_checkpoint=(resume_mode != 'fresh'),
        num_epochs=num_epochs
    )

def create_academic_report(dataset_key, experiment_id=None):
    """
    Convenience function for creating academic reports.

    Args:
        dataset_key (str): 'mnist' or 'cifar10'
        experiment_id (str): Optional experiment ID

    Returns:
        tuple: (reporter, report_path)
    """
    if not _imports_successful:
        raise ImportError("Required modules not available")

    return run_fixed_fully_integrated_academic_study(
        dataset_choice=dataset_key,
        experiment_id=experiment_id
    )

# Package info
def get_info():
    """Get package information."""
    info = {
        'version': __version__,
        'author': __author__,
        'imports_successful': _imports_successful,
    }

    if _imports_successful:
        info.update({
            'device': str(device),
            'device_name': device_name,
            'device_type': device_type,
            'available_datasets': list(DATASETS.keys()),
        })

    return info

# Expose main functionality
__all__ = [
    '__version__',
    'train_enhanced_gan',
    'create_academic_report',
    'get_info',
]

if _imports_successful:
    __all__.extend([
        'EnhancedConditionalGenerator',
        'EnhancedConditionalCritic',
        'EMAGenerator',
        'WassersteinGPLoss',
        'FixedFullyIntegratedAcademicReporter',
        'InteractiveDigitGenerator',
        'DATASETS',
        'device',
        'device_name',
        'device_type',
    ])