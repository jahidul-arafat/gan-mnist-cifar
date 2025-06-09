#!/usr/bin/env python3
"""
Enhanced DCGAN API Client
========================

Python client for interacting with the Enhanced DCGAN Web API.
Provides easy-to-use methods for training, generation, and monitoring.

File: backend/api_client.py
"""

import requests
import json
import time
import websocket
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import base64
from pathlib import Path

class DCGANApiClient:
    """
    Python client for Enhanced DCGAN Web API
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the API client

        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.ws = None
        self.ws_callbacks = {}

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise DCGANApiError(f"Request failed: {str(e)}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and capabilities

        Returns:
            Dictionary containing system information
        """
        response = self._make_request("GET", "/api/system/status")
        return response.json()

    def get_datasets(self) -> Dict[str, Any]:
        """
        Get available datasets with detailed information

        Returns:
            Dictionary of dataset configurations
        """
        response = self._make_request("GET", "/api/datasets")
        return response.json()

    def get_checkpoints(self, dataset: str) -> List[Dict[str, Any]]:
        """
        Get available checkpoints for a specific dataset

        Args:
            dataset: Dataset name ('mnist' or 'cifar10')

        Returns:
            List of checkpoint information
        """
        response = self._make_request("GET", f"/api/checkpoints/{dataset}")
        return response.json()

    def start_training(self, dataset: str, epochs: int,
                       resume_mode: str = "interactive",
                       experiment_name: Optional[str] = None) -> Dict[str, str]:
        """
        Start GAN training

        Args:
            dataset: Dataset to train on ('mnist' or 'cifar10')
            epochs: Number of training epochs
            resume_mode: Resume mode ('interactive', 'latest', 'fresh')
            experiment_name: Optional experiment name

        Returns:
            Dictionary with training_id and status
        """
        data = {
            "dataset": dataset,
            "epochs": epochs,
            "resume_mode": resume_mode
        }

        if experiment_name:
            data["experiment_name"] = experiment_name

        response = self._make_request("POST", "/api/training/start", json=data)
        return response.json()

    def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """
        Get current status of a training session

        Args:
            training_id: Training session ID

        Returns:
            Training status information
        """
        response = self._make_request("GET", f"/api/training/status/{training_id}")
        return response.json()

    def stop_training(self, training_id: str) -> Dict[str, str]:
        """
        Stop a running training session

        Args:
            training_id: Training session ID

        Returns:
            Status message
        """
        response = self._make_request("POST", f"/api/training/stop/{training_id}")
        return response.json()

    def generate_images(self, prompt: str, dataset: str,
                        num_samples: int = 8, use_ema: bool = True) -> Dict[str, Any]:
        """
        Generate images based on text prompt

        Args:
            prompt: Text prompt describing what to generate
            dataset: Dataset the model was trained on
            num_samples: Number of images to generate
            use_ema: Whether to use EMA generator

        Returns:
            Generation results with image data
        """
        data = {
            "prompt": prompt,
            "dataset": dataset,
            "num_samples": num_samples,
            "use_ema": use_ema
        }

        response = self._make_request("POST", "/api/generate", json=data)
        return response.json()

    def generate_report(self, dataset: str,
                        experiment_id: Optional[str] = None,
                        include_images: bool = True) -> Dict[str, str]:
        """
        Generate academic research report

        Args:
            dataset: Dataset for the report
            experiment_id: Optional specific experiment ID
            include_images: Whether to include generated images

        Returns:
            Report generation status
        """
        data = {
            "dataset": dataset,
            "include_images": include_images
        }

        if experiment_id:
            data["experiment_id"] = experiment_id

        response = self._make_request("POST", "/api/reports/generate", json=data)
        return response.json()

    def get_report(self, report_id: str) -> Dict[str, Any]:
        """
        Get generated report information

        Args:
            report_id: Report ID

        Returns:
            Report information and status
        """
        response = self._make_request("GET", f"/api/reports/{report_id}")
        return response.json()

    def get_training_logs(self, dataset: str) -> List[Dict[str, Any]]:
        """
        Get training logs for a specific dataset

        Args:
            dataset: Dataset name

        Returns:
            List of training log entries
        """
        response = self._make_request("GET", f"/api/logs/{dataset}")
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status

        Returns:
            Health status information
        """
        response = self._make_request("GET", "/health")
        return response.json()

    # WebSocket Methods

    def connect_websocket(self, on_message: Optional[Callable] = None,
                          on_error: Optional[Callable] = None,
                          on_close: Optional[Callable] = None):
        """
        Connect to WebSocket for real-time updates

        Args:
            on_message: Callback for messages
            on_error: Callback for errors
            on_close: Callback for connection close
        """
        ws_url = self.base_url.replace('http', 'ws') + '/ws'

        def on_ws_message(ws, message):
            try:
                data = json.loads(message)
                if on_message:
                    on_message(data)

                # Handle specific message types
                msg_type = data.get('type')
                if msg_type in self.ws_callbacks:
                    self.ws_callbacks[msg_type](data)

            except json.JSONDecodeError as e:
                if on_error:
                    on_error(f"JSON decode error: {e}")

        def on_ws_error(ws, error):
            if on_error:
                on_error(error)

        def on_ws_close(ws, close_status_code, close_msg):
            if on_close:
                on_close(close_status_code, close_msg)

        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_ws_message,
            on_error=on_ws_error,
            on_close=on_ws_close
        )

        # Start WebSocket in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

    def register_callback(self, message_type: str, callback: Callable):
        """
        Register callback for specific WebSocket message types

        Args:
            message_type: Type of message to listen for
            callback: Function to call when message is received
        """
        self.ws_callbacks[message_type] = callback

    def disconnect_websocket(self):
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()
            self.ws = None

    # Convenience Methods

    def wait_for_training_completion(self, training_id: str,
                                     poll_interval: int = 5,
                                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Wait for training to complete, polling for status updates

        Args:
            training_id: Training session ID
            poll_interval: Seconds between status polls
            progress_callback: Optional callback for progress updates

        Returns:
            Final training status
        """
        while True:
            status = self.get_training_status(training_id)

            if progress_callback:
                progress_callback(status)

            if status.get('status') in ['completed', 'error', 'stopped']:
                return status

            time.sleep(poll_interval)

    def train_and_wait(self, dataset: str, epochs: int,
                       resume_mode: str = "interactive",
                       experiment_name: Optional[str] = None,
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Start training and wait for completion

        Args:
            dataset: Dataset to train on
            epochs: Number of epochs
            resume_mode: Resume mode
            experiment_name: Optional experiment name
            progress_callback: Optional progress callback

        Returns:
            Final training status
        """
        result = self.start_training(dataset, epochs, resume_mode, experiment_name)
        training_id = result['training_id']

        return self.wait_for_training_completion(training_id, progress_callback=progress_callback)

    def interactive_generation_session(self, dataset: str):
        """
        Start an interactive generation session

        Args:
            dataset: Dataset for generation

        Returns:
            InteractiveSession object
        """
        return InteractiveSession(self, dataset)

class InteractiveSession:
    """Interactive generation session"""

    def __init__(self, client: DCGANApiClient, dataset: str):
        self.client = client
        self.dataset = dataset
        self.generation_history = []

    def generate(self, prompt: str, num_samples: int = 8) -> Dict[str, Any]:
        """Generate images with prompt"""
        result = self.client.generate_images(prompt, self.dataset, num_samples)
        self.generation_history.append({
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """Get generation history"""
        return self.generation_history

    def save_history(self, filepath: str):
        """Save generation history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.generation_history, f, indent=2)

class DCGANApiError(Exception):
    """Exception raised for API errors"""
    pass

# Example usage and demos
def demo_basic_usage():
    """Demonstrate basic API usage"""
    print("🎨 Enhanced DCGAN API Client Demo")
    print("=" * 50)

    # Initialize client
    client = DCGANApiClient()

    try:
        # Check system status
        print("📊 Checking system status...")
        status = client.get_system_status()
        print(f"   Device: {status['device_name']} ({status['device_type']})")
        print(f"   DCGAN Available: {status['dcgan_available']}")

        # Get available datasets
        print("\n📁 Available datasets:")
        datasets = client.get_datasets()
        for name, info in datasets.items():
            print(f"   {name}: {info['name']} ({info['available_checkpoints']} checkpoints)")

        # Health check
        print("\n🏥 Health check:")
        health = client.health_check()
        print(f"   Status: {health['status']}")

    except DCGANApiError as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def demo_training_session():
    """Demonstrate training session management"""
    print("🚀 Training Session Demo")
    print("=" * 50)

    client = DCGANApiClient()

    def progress_callback(status):
        progress = status.get('progress_percentage', 0)
        epoch = status.get('current_epoch', 0)
        total = status.get('total_epochs', 0)
        print(f"   Progress: {progress:.1f}% (Epoch {epoch}/{total})")

    try:
        # Start training
        print("🎯 Starting MNIST training...")
        result = client.start_training('mnist', 10, 'fresh', 'demo_experiment')
        training_id = result['training_id']
        print(f"   Training ID: {training_id}")

        # Monitor progress (demo - would normally wait for completion)
        print("📊 Monitoring progress...")
        status = client.get_training_status(training_id)
        progress_callback(status)

    except DCGANApiError as e:
        print(f"❌ Training error: {e}")

def demo_interactive_generation():
    """Demonstrate interactive generation"""
    print("🎨 Interactive Generation Demo")
    print("=" * 50)

    client = DCGANApiClient()

    try:
        session = client.interactive_generation_session('mnist')

        prompts = ["Draw me a 7", "Generate 3", "Show me a 9"]

        for prompt in prompts:
            print(f"🎯 Generating: '{prompt}'")
            result = session.generate(prompt, num_samples=4)
            print(f"   Generation ID: {result.get('generation_id', 'N/A')}")

        print(f"\n📋 Session history: {len(session.get_history())} generations")

    except DCGANApiError as e:
        print(f"❌ Generation error: {e}")

if __name__ == "__main__":
    # Run demos
    demo_basic_usage()
    print("\n")
    demo_training_session()
    print("\n")
    demo_interactive_generation()