#!/usr/bin/env python3
"""
Enhanced DCGAN Health Check Script
=================================

Comprehensive health monitoring for the Enhanced DCGAN web backend.
Checks system resources, GPU availability, model integrity, and API endpoints.

File: backend/health_check.py
"""

import os
import sys
import time
import psutil
import requests
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import subprocess

# Try to import torch for GPU checks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class HealthChecker:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.storage_root = Path("./storage")
        self.results = {}

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system CPU, memory, and disk usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "status": "healthy",
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "warnings": self._generate_resource_warnings(cpu_percent, memory.percent, disk.percent)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _generate_resource_warnings(self, cpu: float, memory: float, disk: float) -> List[str]:
        """Generate warnings for resource usage"""
        warnings = []
        if cpu > 90:
            warnings.append("High CPU usage detected")
        if memory > 90:
            warnings.append("High memory usage detected")
        if disk > 90:
            warnings.append("High disk usage detected")
        return warnings

    def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and CUDA status"""
        if not TORCH_AVAILABLE:
            return {
                "status": "unavailable",
                "error": "PyTorch not available"
            }

        try:
            gpu_info = {
                "status": "available" if torch.cuda.is_available() else "unavailable",
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                "devices": []
            }

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    memory_total = device_props.total_memory / (1024**3)

                    gpu_info["devices"].append({
                        "device_id": i,
                        "name": device_props.name,
                        "memory_total_gb": memory_total,
                        "memory_allocated_gb": memory_allocated,
                        "memory_reserved_gb": memory_reserved,
                        "memory_free_gb": memory_total - memory_reserved,
                        "compute_capability": f"{device_props.major}.{device_props.minor}"
                    })

            # Check for Apple MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info["mps_available"] = True
                gpu_info["mps_built"] = torch.backends.mps.is_built()

            return gpu_info

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def check_storage_directories(self) -> Dict[str, Any]:
        """Check storage directories and permissions"""
        directories = [
            self.storage_root / "models",
            self.storage_root / "reports",
            self.storage_root / "static",
            self.storage_root / "training_logs"
        ]

        dir_status = {}
        all_healthy = True

        for directory in directories:
            try:
                exists = directory.exists()
                writable = os.access(directory, os.W_OK) if exists else False
                readable = os.access(directory, os.R_OK) if exists else False

                if exists:
                    files_count = len(list(directory.glob('*')))
                    size_mb = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file()) / (1024**2)
                else:
                    files_count = 0
                    size_mb = 0

                dir_healthy = exists and writable and readable
                if not dir_healthy:
                    all_healthy = False

                dir_status[str(directory)] = {
                    "exists": exists,
                    "writable": writable,
                    "readable": readable,
                    "files_count": files_count,
                    "size_mb": round(size_mb, 2),
                    "healthy": dir_healthy
                }

            except Exception as e:
                dir_status[str(directory)] = {
                    "error": str(e),
                    "healthy": False
                }
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "directories": dir_status
        }

    def check_api_endpoints(self) -> Dict[str, Any]:
        """Check API endpoint availability and response times"""
        endpoints = [
            "/",
            "/health",
            "/api/system/status",
            "/api/datasets"
        ]

        endpoint_status = {}
        all_healthy = True

        for endpoint in endpoints:
            url = f"{self.api_url}{endpoint}"
            try:
                start_time = time.time()
                response = requests.get(url, timeout=10)
                response_time = time.time() - start_time

                endpoint_healthy = response.status_code == 200
                if not endpoint_healthy:
                    all_healthy = False

                endpoint_status[endpoint] = {
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "healthy": endpoint_healthy,
                    "content_type": response.headers.get("content-type", "unknown")
                }

                # Try to parse JSON response
                try:
                    json_data = response.json()
                    endpoint_status[endpoint]["response_valid"] = True
                    endpoint_status[endpoint]["response_size"] = len(str(json_data))
                except:
                    endpoint_status[endpoint]["response_valid"] = False

            except requests.RequestException as e:
                endpoint_status[endpoint] = {
                    "error": str(e),
                    "healthy": False
                }
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "endpoints": endpoint_status
        }

    def check_dcgan_modules(self) -> Dict[str, Any]:
        """Check Enhanced DCGAN module availability"""
        try:
            # Try to import Enhanced DCGAN modules
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))

            from enhanced_dcgan_research import (
                DATASETS, device, device_name, device_type,
                find_available_checkpoints, EnhancedConditionalGenerator
            )

            module_status = {
                "status": "available",
                "datasets_available": list(DATASETS.keys()),
                "device_name": device_name,
                "device_type": device_type,
                "checkpoints": {}
            }

            # Check checkpoints for each dataset
            for dataset_key in DATASETS.keys():
                checkpoints = find_available_checkpoints(dataset_key)
                module_status["checkpoints"][dataset_key] = len(checkpoints)

            # Test model initialization
            try:
                test_generator = EnhancedConditionalGenerator(100, 10, 1)
                module_status["model_initialization"] = "success"
            except Exception as e:
                module_status["model_initialization"] = f"failed: {str(e)}"

            return module_status

        except ImportError as e:
            return {
                "status": "unavailable",
                "error": f"Import error: {str(e)}",
                "suggestion": "Ensure enhanced_dcgan_research package is properly installed"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def check_dependencies(self) -> Dict[str, Any]:
        """Check critical Python dependencies"""
        critical_packages = [
            "fastapi", "uvicorn", "torch", "torchvision",
            "numpy", "pandas", "matplotlib", "pillow", "tqdm"
        ]

        package_status = {}
        all_available = True

        for package in critical_packages:
            try:
                __import__(package)
                # Get version if possible
                try:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                except:
                    version = 'unknown'

                package_status[package] = {
                    "available": True,
                    "version": version
                }
            except ImportError:
                package_status[package] = {
                    "available": False,
                    "version": None
                }
                all_available = False

        return {
            "status": "healthy" if all_available else "unhealthy",
            "packages": package_status,
            "missing_packages": [pkg for pkg, status in package_status.items() if not status["available"]]
        }

    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all health checks and compile results"""
        print("🔍 Running Enhanced DCGAN Health Check...")

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {
                "system_resources": self.check_system_resources(),
                "gpu_availability": self.check_gpu_availability(),
                "storage_directories": self.check_storage_directories(),
                "api_endpoints": self.check_api_endpoints(),
                "dcgan_modules": self.check_dcgan_modules(),
                "dependencies": self.check_dependencies()
            }
        }

        # Determine overall status
        failed_checks = []
        for check_name, check_result in self.results["checks"].items():
            if check_result.get("status") not in ["healthy", "available"]:
                failed_checks.append(check_name)

        if failed_checks:
            self.results["overall_status"] = "unhealthy"
            self.results["failed_checks"] = failed_checks

        return self.results

    def print_results(self):
        """Print formatted health check results"""
        if not self.results:
            print("❌ No health check results available")
            return

        print("\n" + "="*80)
        print("🏥 ENHANCED DCGAN HEALTH CHECK RESULTS")
        print("="*80)

        # Overall status
        status_emoji = "✅" if self.results["overall_status"] == "healthy" else "❌"
        print(f"\n{status_emoji} Overall Status: {self.results['overall_status'].upper()}")
        print(f"🕒 Timestamp: {self.results['timestamp']}")

        # Individual checks
        for check_name, check_result in self.results["checks"].items():
            print(f"\n📋 {check_name.replace('_', ' ').title()}:")

            status = check_result.get("status", "unknown")
            status_emoji = "✅" if status in ["healthy", "available"] else "❌"
            print(f"   {status_emoji} Status: {status}")

            # Print key details for each check
            if check_name == "system_resources":
                print(f"   💻 CPU Usage: {check_result.get('cpu_usage_percent', 0):.1f}%")
                print(f"   🧠 Memory Usage: {check_result.get('memory_usage_percent', 0):.1f}%")
                print(f"   💾 Disk Usage: {check_result.get('disk_usage_percent', 0):.1f}%")

            elif check_name == "gpu_availability":
                if check_result.get("cuda_available"):
                    print(f"   🎮 CUDA Devices: {check_result.get('device_count', 0)}")
                    for device in check_result.get("devices", []):
                        print(f"      GPU {device['device_id']}: {device['name']} ({device['memory_total_gb']:.1f}GB)")

            elif check_name == "dcgan_modules":
                if status == "available":
                    print(f"   📊 Datasets: {', '.join(check_result.get('datasets_available', []))}")
                    print(f"   🖥️  Device: {check_result.get('device_name', 'Unknown')}")

            elif check_name == "dependencies":
                missing = check_result.get("missing_packages", [])
                if missing:
                    print(f"   ⚠️  Missing packages: {', '.join(missing)}")

            # Print errors if any
            if "error" in check_result:
                print(f"   ❌ Error: {check_result['error']}")

        # Recommendations
        if self.results["overall_status"] != "healthy":
            print(f"\n💡 RECOMMENDATIONS:")
            failed_checks = self.results.get("failed_checks", [])

            for check in failed_checks:
                if check == "gpu_availability":
                    print("   • Install CUDA drivers and PyTorch with CUDA support")
                elif check == "dcgan_modules":
                    print("   • Install the enhanced_dcgan_research package")
                elif check == "dependencies":
                    print("   • Install missing Python packages: pip install -r requirements.txt")
                elif check == "storage_directories":
                    print("   • Create storage directories with proper permissions")
                elif check == "api_endpoints":
                    print("   • Start the FastAPI backend server")

        print("="*80)

    def save_results(self, filepath: str = None):
        """Save health check results to JSON file"""
        if not self.results:
            print("❌ No results to save")
            return

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"health_check_{timestamp}.json"

        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"✅ Health check results saved to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    """Main health check execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced DCGAN Health Check")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="API URL to check (default: http://localhost:8000)")
    parser.add_argument("--save", help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Only show overall status")

    args = parser.parse_args()

    # Run health check
    checker = HealthChecker(api_url=args.api_url)
    results = checker.run_comprehensive_check()

    if args.quiet:
        print(f"Health Status: {results['overall_status']}")
        sys.exit(0 if results['overall_status'] == 'healthy' else 1)

    # Print detailed results
    checker.print_results()

    # Save results if requested
    if args.save:
        checker.save_results(args.save)

    # Exit with appropriate code
    exit_code = 0 if results['overall_status'] == 'healthy' else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()