# File: backend/training_integration.py
"""
Training Integration with WebSocket Updates
==========================================

This script modifies your existing training to send real-time updates
to the frontend via WebSocket connections.
"""

import asyncio
import threading
import re
from datetime import datetime
from typing import Optional, Dict, Any
from websocket_manager import websocket_manager

class TrainingWebSocketIntegration:
    """
    Integration class to add WebSocket updates to existing training
    """

    def __init__(self, training_id: str, dataset: str):
        self.training_id = training_id
        self.dataset = dataset
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.metrics = {}

        # Track resume information
        self.resumed_from_epoch = 0
        self.is_resumed_training = False
        self.session_start_epoch = 0  # The epoch this session started from

    def set_resume_info(self, resumed_from_epoch: int):
        """Set information about resumed training"""
        self.resumed_from_epoch = resumed_from_epoch
        self.is_resumed_training = resumed_from_epoch > 0
        self.session_start_epoch = resumed_from_epoch

        print(f"🔄 Training resumed from epoch {resumed_from_epoch}")

    def calculate_absolute_epoch(self, relative_epoch: int) -> int:
        """Calculate absolute epoch number from relative epoch"""
        if self.is_resumed_training:
            # For resumed training: absolute = resumed_from + relative
            absolute_epoch = self.resumed_from_epoch + relative_epoch
        else:
            # For fresh training: absolute = relative
            absolute_epoch = relative_epoch

        return absolute_epoch

    def calculate_progress_percentage(self, absolute_epoch: int) -> float:
        """Calculate correct progress percentage based on absolute epoch"""
        if self.total_epochs > 0:
            return (absolute_epoch / self.total_epochs) * 100
        return 0

    async def notify_training_start(self, total_epochs: int, resumed_from_epoch: int = 0):
        """Notify that training has started"""
        self.total_epochs = total_epochs
        self.set_resume_info(resumed_from_epoch)

        initial_epoch = resumed_from_epoch if resumed_from_epoch > 0 else 0
        progress = self.calculate_progress_percentage(initial_epoch)

        await websocket_manager.send_training_update(self.training_id, {
            "status": "running",
            "dataset": self.dataset,
            "current_epoch": initial_epoch,
            "total_epochs": total_epochs,
            "progress_percentage": progress,
            "resumed_from_epoch": resumed_from_epoch,
            "is_resumed": self.is_resumed_training,
            "timestamp": datetime.now().isoformat()
        })

        resume_msg = f" (resumed from epoch {resumed_from_epoch})" if self.is_resumed_training else ""
        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Training started for {self.dataset.upper()} - {total_epochs} epochs{resume_msg}",
            "dataset": self.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    async def notify_epoch_update(self, relative_epoch: int, total_epochs: int = None):
        """Notify epoch progress with correct absolute numbers"""
        if total_epochs:
            self.total_epochs = total_epochs

        # Calculate absolute epoch number
        absolute_epoch = self.calculate_absolute_epoch(relative_epoch)
        progress = self.calculate_progress_percentage(absolute_epoch)

        self.current_epoch = absolute_epoch

        await websocket_manager.send_training_update(self.training_id, {
            "status": "running",
            "dataset": self.dataset,
            "current_epoch": absolute_epoch,
            "total_epochs": self.total_epochs,
            "progress_percentage": progress,
            "relative_epoch": relative_epoch,  # Keep for debugging
            "resumed_from_epoch": self.resumed_from_epoch,
            "is_resumed": self.is_resumed_training,
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Epoch {absolute_epoch}/{self.total_epochs} in progress ({progress:.1f}%)",
            "dataset": self.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    async def notify_batch_update(self, relative_epoch: int, batch: int, total_batches: int, metrics: Dict[str, Any]):
        """Notify batch completion with correct epoch numbers"""
        absolute_epoch = self.calculate_absolute_epoch(relative_epoch)
        progress = self.calculate_progress_percentage(absolute_epoch)

        self.current_epoch = absolute_epoch
        self.current_batch = batch
        self.total_batches = total_batches
        self.metrics = metrics

        # Only send updates every 25 batches to avoid overwhelming
        if batch % 25 == 0 or batch == total_batches:
            batch_progress = (batch / total_batches) * 100 if total_batches > 0 else 0

            await websocket_manager.send_training_update(self.training_id, {
                "status": "running",
                "dataset": self.dataset,
                "current_epoch": absolute_epoch,
                "total_epochs": self.total_epochs,
                "current_batch": batch,
                "total_batches": total_batches,
                "batch_progress": batch_progress,
                "progress_percentage": progress,
                "metrics": metrics,
                "relative_epoch": relative_epoch,  # Keep for debugging
                "resumed_from_epoch": self.resumed_from_epoch,
                "is_resumed": self.is_resumed_training,
                "timestamp": datetime.now().isoformat()
            })

            # Extract key metrics for logging
            g_loss = metrics.get('generator_loss', metrics.get('g_loss', 0))
            d_loss = metrics.get('discriminator_loss', metrics.get('d_loss', 0))

            await websocket_manager.send_log_message({
                "level": "debug",
                "message": f"Epoch {absolute_epoch}/{self.total_epochs}, Batch {batch}/{total_batches} - G_Loss: {g_loss:.4f}, D_Loss: {d_loss:.4f}",
                "dataset": self.dataset,
                "source": "training",
                "timestamp": datetime.now().isoformat()
            })

    async def notify_epoch_complete(self, relative_epoch: int, metrics: Dict[str, Any]):
        """Notify that an epoch has completed"""
        absolute_epoch = self.calculate_absolute_epoch(relative_epoch)
        progress = self.calculate_progress_percentage(absolute_epoch)

        await websocket_manager.send_training_update(self.training_id, {
            "status": "running",
            "dataset": self.dataset,
            "current_epoch": absolute_epoch,
            "total_epochs": self.total_epochs,
            "progress_percentage": progress,
            "metrics": metrics,
            "relative_epoch": relative_epoch,
            "resumed_from_epoch": self.resumed_from_epoch,
            "is_resumed": self.is_resumed_training,
            "timestamp": datetime.now().isoformat()
        })

        # Log epoch completion
        g_loss = metrics.get('generator_loss', metrics.get('g_loss', 0))
        d_loss = metrics.get('discriminator_loss', metrics.get('d_loss', 0))

        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Epoch {absolute_epoch}/{self.total_epochs} completed - G_Loss: {g_loss:.4f}, D_Loss: {d_loss:.4f}",
            "dataset": self.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    async def notify_checkpoint_saved(self, relative_epoch: int, checkpoint_path: str):
        """Notify that a checkpoint has been saved"""
        absolute_epoch = self.calculate_absolute_epoch(relative_epoch)

        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Checkpoint saved at epoch {absolute_epoch}: {checkpoint_path}",
            "dataset": self.dataset,
            "source": "checkpoint",
            "timestamp": datetime.now().isoformat()
        })

    async def notify_training_complete(self, final_metrics: Dict[str, Any]):
        """Notify that training has completed successfully"""
        await websocket_manager.send_training_update(self.training_id, {
            "status": "completed",
            "dataset": self.dataset,
            "current_epoch": self.total_epochs,
            "total_epochs": self.total_epochs,
            "progress_percentage": 100,
            "metrics": final_metrics,
            "resumed_from_epoch": self.resumed_from_epoch,
            "is_resumed": self.is_resumed_training,
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Training completed successfully for {self.dataset.upper()}",
            "dataset": self.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    async def notify_training_error(self, error_message: str):
        """Notify that training has encountered an error"""
        await websocket_manager.send_training_update(self.training_id, {
            "status": "error",
            "dataset": self.dataset,
            "error_message": error_message,
            "resumed_from_epoch": self.resumed_from_epoch,
            "is_resumed": self.is_resumed_training,
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "error",
            "message": f"Training error: {error_message}",
            "dataset": self.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

    async def notify_training_stopped(self):
        """Notify that training has been stopped by user"""
        await websocket_manager.send_training_update(self.training_id, {
            "status": "stopped",
            "dataset": self.dataset,
            "resumed_from_epoch": self.resumed_from_epoch,
            "is_resumed": self.is_resumed_training,
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "warning",
            "message": f"Training stopped by user for {self.dataset.upper()}",
            "dataset": self.dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })


class TrainingProgressParser:
    """
    Parser to extract training progress from console output and convert to WebSocket updates
    """

    def __init__(self, training_id: str, dataset: str):
        self.websocket_integration = TrainingWebSocketIntegration(training_id, dataset)
        self.training_started = False

    def parse_and_send_update(self, console_line: str):
        """Parse console output line and send appropriate WebSocket update"""
        try:
            # Parse resumed training info
            resumed_match = re.search(r'🔄 RESUMED from checkpoint at epoch (\d+)', console_line)
            if resumed_match:
                resumed_epoch = int(resumed_match.group(1))
                self.websocket_integration.set_resume_info(resumed_epoch)
                asyncio.create_task(self.websocket_integration.notify_training_start(50, resumed_epoch))
                return

            # Parse epoch progress: "EPOCH 27/50" or "Epoch 27/50"
            epoch_match = re.search(r'(?:EPOCH|Epoch)\s+(\d+)/(\d+)', console_line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))

                # If this is a resumed training, the current_epoch is already absolute
                # If it's fresh training, we need to handle it differently
                if self.websocket_integration.is_resumed_training:
                    # For resumed training, the epoch numbers in console are absolute
                    relative_epoch = current_epoch - self.websocket_integration.resumed_from_epoch
                else:
                    # For fresh training, epoch numbers are relative
                    relative_epoch = current_epoch

                asyncio.create_task(
                    self.websocket_integration.notify_epoch_update(relative_epoch, total_epochs)
                )
                return

            # Parse batch progress with metrics
            batch_match = re.search(r'(\d+)/(\d+).*D[_\s]*Loss[:\s]*([+-]?\d+\.?\d*)', console_line)
            if batch_match:
                batch = int(batch_match.group(1))
                total_batches = int(batch_match.group(2))
                d_loss = float(batch_match.group(3))

                # Extract generator loss if present
                g_loss_match = re.search(r'G[_\s]*Loss[:\s]*([+-]?\d+\.?\d*)', console_line)
                g_loss = float(g_loss_match.group(1)) if g_loss_match else 0

                metrics = {
                    'discriminator_loss': d_loss,
                    'generator_loss': g_loss
                }

                # Current epoch from the websocket integration
                current_relative_epoch = self.websocket_integration.current_epoch - self.websocket_integration.resumed_from_epoch if self.websocket_integration.is_resumed_training else self.websocket_integration.current_epoch

                asyncio.create_task(
                    self.websocket_integration.notify_batch_update(
                        current_relative_epoch, batch, total_batches, metrics
                    )
                )
                return

        except Exception as e:
            print(f"Error parsing training progress: {e}")


def run_async_in_thread(coro):
    """Helper function to run async functions in a synchronous context"""
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()


# Example integration with your existing training code
class EnhancedTrainingWrapper:
    """
    Wrapper class to integrate WebSocket updates with your existing training
    """

    def __init__(self, training_id: str, dataset: str):
        self.websocket_integration = TrainingWebSocketIntegration(training_id, dataset)

    def start_training(self, total_epochs: int, resumed_from_epoch: int = 0):
        """Call this at the start of training"""
        run_async_in_thread(
            self.websocket_integration.notify_training_start(total_epochs, resumed_from_epoch)
        )

    def on_epoch_progress(self, relative_epoch: int, total_epochs: int = None):
        """Call this during epoch progress (relative to the current training session)"""
        run_async_in_thread(
            self.websocket_integration.notify_epoch_update(relative_epoch, total_epochs)
        )

    def on_batch_complete(self, relative_epoch: int, batch: int, total_batches: int, metrics: dict):
        """Call this after each batch (with relative epoch)"""
        run_async_in_thread(
            self.websocket_integration.notify_batch_update(relative_epoch, batch, total_batches, metrics)
        )

    def on_epoch_complete(self, relative_epoch: int, metrics: dict):
        """Call this at the end of each epoch (with relative epoch)"""
        run_async_in_thread(
            self.websocket_integration.notify_epoch_complete(relative_epoch, metrics)
        )

    def on_checkpoint_saved(self, relative_epoch: int, checkpoint_path: str):
        """Call this when a checkpoint is saved (with relative epoch)"""
        run_async_in_thread(
            self.websocket_integration.notify_checkpoint_saved(relative_epoch, checkpoint_path)
        )

    def on_training_complete(self, final_metrics: dict):
        """Call this when training completes successfully"""
        run_async_in_thread(
            self.websocket_integration.notify_training_complete(final_metrics)
        )

    def on_training_error(self, error_message: str):
        """Call this when training encounters an error"""
        run_async_in_thread(
            self.websocket_integration.notify_training_error(error_message)
        )

    def on_training_stopped(self):
        """Call this when training is stopped by user"""
        run_async_in_thread(
            self.websocket_integration.notify_training_stopped()
        )


# Quick integration example for your existing training function
def integrate_with_existing_training():
    """
    Example of how to integrate with your existing training code

    Add these lines to your existing training function:
    """

    # At the start of your training function:
    # training_wrapper = EnhancedTrainingWrapper(training_id, dataset_name)
    # training_wrapper.start_training(num_epochs, resumed_from_epoch)

    # In your epoch loop:
    # training_wrapper.on_epoch_progress(relative_epoch)

    # In your batch loop:
    # training_wrapper.on_batch_complete(relative_epoch, batch_idx, total_batches, {
    #     'generator_loss': g_loss,
    #     'discriminator_loss': d_loss,
    #     'wasserstein_distance': w_dist
    # })

    # At epoch end:
    # training_wrapper.on_epoch_complete(relative_epoch, epoch_metrics)

    # When saving checkpoints:
    # training_wrapper.on_checkpoint_saved(relative_epoch, checkpoint_path)

    # At training completion:
    # training_wrapper.on_training_complete(final_metrics)

    pass