# File: backend/training_wrapper.py
"""
Non-Interactive Training Wrapper
================================

Wrapper to run Enhanced DCGAN training without interactive prompts
in web backend environment.
"""

import sys
import os
import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

# Add path to enhanced_dcgan_research
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from enhanced_dcgan_research import (
        DATASETS,
        device, device_name, device_type,
        find_available_checkpoints,
        train_enhanced_gan_with_resume_modified,
    )
    DCGAN_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced DCGAN not available: {e}")
    DCGAN_AVAILABLE = False

from websocket_manager import websocket_manager

class NonInteractiveTrainingWrapper:
    """
    Wrapper to run Enhanced DCGAN training without interactive input
    """

    def __init__(self, training_id: str):
        self.training_id = training_id

    def select_checkpoint_automatically(self, dataset_key: str, resume_mode: str) -> Optional[str]:
        """
        Automatically select checkpoint based on resume mode
        """
        try:
            checkpoints = find_available_checkpoints(dataset_key)

            if not checkpoints:
                print(f"No checkpoints found for {dataset_key}")
                return None

            if resume_mode == 'fresh':
                return None

            elif resume_mode == 'latest':
                # Find the most recent checkpoint
                latest_checkpoint = None
                latest_time = 0

                for checkpoint in checkpoints:
                    try:
                        mtime = os.path.getmtime(checkpoint)
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_checkpoint = checkpoint
                    except:
                        continue

                print(f"Auto-selected latest checkpoint: {latest_checkpoint}")
                return latest_checkpoint

            elif resume_mode == 'interactive':
                # In non-interactive mode, prefer 'best' or 'final' models
                for checkpoint in checkpoints:
                    filename = os.path.basename(checkpoint)
                    if 'best' in filename.lower():
                        print(f"Auto-selected best checkpoint: {checkpoint}")
                        return checkpoint

                for checkpoint in checkpoints:
                    filename = os.path.basename(checkpoint)
                    if 'final' in filename.lower():
                        print(f"Auto-selected final checkpoint: {checkpoint}")
                        return checkpoint

                # Fallback to latest
                return self.select_checkpoint_automatically(dataset_key, 'latest')

            else:
                print(f"Unknown resume mode: {resume_mode}")
                return None

        except Exception as e:
            print(f"Error selecting checkpoint: {e}")
            return None

    async def run_training_non_interactive(self, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Run training without interactive prompts
        """
        dataset_key = config['dataset']
        resume_mode = config.get('resume_mode', 'interactive')
        num_epochs = config.get('epochs', 50)
        experiment_name = config.get('experiment_name')

        # Send initial status
        await websocket_manager.send_training_update(self.training_id, {
            "status": "initializing",
            "dataset": dataset_key,
            "total_epochs": num_epochs,
            "message": "Preparing training environment...",
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Initializing {dataset_key.upper()} training with {num_epochs} epochs",
            "dataset": dataset_key,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

        try:
            # Check if dataset exists
            if dataset_key not in DATASETS:
                raise ValueError(f"Unknown dataset: {dataset_key}")

            dataset_config = DATASETS[dataset_key]

            # Handle checkpoint selection automatically
            if resume_mode != 'fresh':
                await websocket_manager.send_log_message({
                    "level": "info",
                    "message": f"Searching for checkpoints (mode: {resume_mode})...",
                    "dataset": dataset_key,
                    "source": "checkpoint",
                    "timestamp": datetime.now().isoformat()
                })

                selected_checkpoint = self.select_checkpoint_automatically(dataset_key, resume_mode)

                if selected_checkpoint:
                    await websocket_manager.send_log_message({
                        "level": "info",
                        "message": f"Auto-selected checkpoint: {os.path.basename(selected_checkpoint)}",
                        "dataset": dataset_key,
                        "source": "checkpoint",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    await websocket_manager.send_log_message({
                        "level": "warning",
                        "message": "No suitable checkpoint found, starting fresh training",
                        "dataset": dataset_key,
                        "source": "checkpoint",
                        "timestamp": datetime.now().isoformat()
                    })
                    resume_mode = 'fresh'

            # Update status to starting
            await websocket_manager.send_training_update(self.training_id, {
                "status": "starting",
                "dataset": dataset_key,
                "total_epochs": num_epochs,
                "message": "Starting training...",
                "timestamp": datetime.now().isoformat()
            })

            # Monkey patch the interactive functions to be non-interactive
            self.patch_interactive_functions(dataset_key, resume_mode)

            # Run the actual training
            print(f"🚀 Starting training for {dataset_key} with {num_epochs} epochs")
            print(f"📊 Resume mode: {resume_mode}")
            print(f"🖥️  Device: {device_name} ({device_type})")

            result = train_enhanced_gan_with_resume_modified(
                dataset_key=dataset_key,
                config=dataset_config,
                resume_from_checkpoint=(resume_mode != 'fresh'),
                num_epochs=num_epochs,
                experiment_name=experiment_name
            )

            await websocket_manager.send_log_message({
                "level": "info",
                "message": f"Training completed successfully for {dataset_key.upper()}",
                "dataset": dataset_key,
                "source": "training",
                "timestamp": datetime.now().isoformat()
            })

            return result

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            print(f"❌ {error_msg}")

            await websocket_manager.send_log_message({
                "level": "error",
                "message": error_msg,
                "dataset": dataset_key,
                "source": "training",
                "timestamp": datetime.now().isoformat()
            })

            raise e

    def patch_interactive_functions(self, dataset_key: str, resume_mode: str):
        """
        Monkey patch interactive functions to be non-interactive
        """
        try:
            # Import the specific module that contains the interactive functions
            import enhanced_dcgan_research.enhanced_dcgan_mnist_cifar_for_apple_mps_checkpoints_graceful_v02 as dcgan_module

            # Store original functions
            original_get_checkpoint_choice = dcgan_module.get_checkpoint_choice
            original_get_dataset_choice = dcgan_module.get_dataset_choice

            def non_interactive_get_checkpoint_choice(dataset_key_param):
                """Non-interactive version of get_checkpoint_choice"""
                print(f"🔄 Non-interactive checkpoint selection for {dataset_key_param}")

                if resume_mode == 'fresh':
                    return None, None

                selected_checkpoint = self.select_checkpoint_automatically(dataset_key_param, resume_mode)

                if selected_checkpoint:
                    print(f"✅ Selected checkpoint: {selected_checkpoint}")
                    # Load checkpoint data
                    try:
                        import torch
                        checkpoint_data = torch.load(selected_checkpoint, map_location='cpu', weights_only=False)
                        return selected_checkpoint, checkpoint_data
                    except Exception as e:
                        print(f"Failed to load checkpoint {selected_checkpoint}: {e}")
                        return None, None
                else:
                    print("No checkpoint selected, starting fresh")
                    return None, None

            def non_interactive_get_dataset_choice():
                """Non-interactive version of get_dataset_choice"""
                print(f"🔄 Non-interactive dataset selection: {dataset_key}")
                return dataset_key

            # Apply patches
            dcgan_module.get_checkpoint_choice = non_interactive_get_checkpoint_choice
            dcgan_module.get_dataset_choice = non_interactive_get_dataset_choice

            print("✅ Applied non-interactive patches")

        except Exception as e:
            print(f"⚠️  Warning: Could not patch interactive functions: {e}")
            print("Training may still prompt for input")

async def run_non_interactive_training(training_id: str, config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Main function to run non-interactive training
    """
    wrapper = NonInteractiveTrainingWrapper(training_id)
    return await wrapper.run_training_non_interactive(config)