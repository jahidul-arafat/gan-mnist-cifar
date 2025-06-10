#!/usr/bin/env python3
"""
Test WebSocket Connection Script
===============================

Quick script to test WebSocket connectivity and send sample messages
to verify frontend integration.

File: backend/test_websocket.py
"""

import asyncio
import json
import uuid
from datetime import datetime
from websocket_manager import websocket_manager

async def send_test_messages():
    """Send test messages to simulate training updates and logs"""

    print("🧪 Starting WebSocket test simulation...")

    # Test training session
    training_id = str(uuid.uuid4())
    dataset = "mnist"
    total_epochs = 5

    # Send training start message
    await websocket_manager.send_training_update(training_id, {
        "status": "starting",
        "dataset": dataset,
        "total_epochs": total_epochs,
        "current_epoch": 0,
        "progress_percentage": 0,
        "timestamp": datetime.now().isoformat()
    })

    await websocket_manager.send_log_message({
        "level": "info",
        "message": f"Test training started for {dataset.upper()} dataset",
        "dataset": dataset,
        "source": "training",
        "timestamp": datetime.now().isoformat()
    })

    # Simulate training epochs
    for epoch in range(1, total_epochs + 1):
        print(f"📊 Simulating epoch {epoch}/{total_epochs}")

        # Send epoch start
        await websocket_manager.send_training_update(training_id, {
            "status": "running",
            "dataset": dataset,
            "current_epoch": epoch,
            "total_epochs": total_epochs,
            "progress_percentage": (epoch / total_epochs) * 100,
            "timestamp": datetime.now().isoformat()
        })

        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Starting epoch {epoch}/{total_epochs}",
            "dataset": dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

        # Simulate batches
        total_batches = 50
        for batch in range(1, total_batches + 1, 10):  # Every 10 batches
            metrics = {
                "generator_loss": max(0.1, 0.8 - (epoch * 0.1) + (batch * 0.001)),
                "discriminator_loss": max(0.1, 0.7 - (epoch * 0.08) + (batch * 0.0008)),
                "wasserstein_distance": -1.5 + (epoch * 0.2)
            }

            await websocket_manager.send_training_update(training_id, {
                "status": "running",
                "dataset": dataset,
                "current_epoch": epoch,
                "total_epochs": total_epochs,
                "current_batch": batch,
                "total_batches": total_batches,
                "batch_progress": (batch / total_batches) * 100,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            })

            await websocket_manager.send_log_message({
                "level": "debug",
                "message": f"Batch {batch}/{total_batches} - G_Loss: {metrics['generator_loss']:.4f}, D_Loss: {metrics['discriminator_loss']:.4f}",
                "dataset": dataset,
                "source": "training",
                "timestamp": datetime.now().isoformat()
            })

            await asyncio.sleep(0.5)  # Small delay

        # Send epoch complete
        await websocket_manager.send_log_message({
            "level": "info",
            "message": f"Epoch {epoch} completed successfully",
            "dataset": dataset,
            "source": "training",
            "timestamp": datetime.now().isoformat()
        })

        await asyncio.sleep(1)  # Delay between epochs

    # Send training complete
    await websocket_manager.send_training_update(training_id, {
        "status": "completed",
        "dataset": dataset,
        "current_epoch": total_epochs,
        "total_epochs": total_epochs,
        "progress_percentage": 100,
        "timestamp": datetime.now().isoformat()
    })

    await websocket_manager.send_log_message({
        "level": "info",
        "message": f"Test training completed successfully for {dataset.upper()}",
        "dataset": dataset,
        "source": "training",
        "timestamp": datetime.now().isoformat()
    })

    print("✅ WebSocket test simulation completed!")

async def send_sample_logs():
    """Send various sample log messages"""

    sample_logs = [
        {
            "level": "info",
            "message": "System initialized successfully",
            "dataset": "system",
            "source": "system"
        },
        {
            "level": "debug",
            "message": "Loading checkpoint: mnist_best_enhanced_model.pth",
            "dataset": "mnist",
            "source": "checkpoint"
        },
        {
            "level": "warning",
            "message": "High GPU memory usage detected: 85%",
            "dataset": "system",
            "source": "system"
        },
        {
            "level": "info",
            "message": "Model weights saved successfully",
            "dataset": "cifar10",
            "source": "checkpoint"
        }
    ]

    for log in sample_logs:
        await websocket_manager.send_log_message(log)
        await asyncio.sleep(1)

if __name__ == "__main__":
    # This script should be run while your FastAPI server is running
    # and frontend is connected to see the messages

    print("🚀 WebSocket Test Script")
    print("📝 Make sure your FastAPI server is running")
    print("🌐 Connect your frontend to see the test messages")
    print("⏳ Starting in 3 seconds...")

    async def main():
        await asyncio.sleep(3)

        print("\n📨 Sending sample logs...")
        await send_sample_logs()

        print("\n🏋️ Starting training simulation...")
        await send_test_messages()

        print("\n✅ Test completed! Check your frontend for updates.")

    asyncio.run(main())