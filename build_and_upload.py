#!/usr/bin/env python3
"""
Script to build and upload package to TestPyPI
"""

import os
import sys
import subprocess
import shutil

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        print(f"Error output: {result.stderr}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

def main():
    """Main build and upload process."""

    print("🚀 Enhanced DCGAN Research Package Build & Upload")
    print("=" * 60)

    # Check if we're in the right directory
    if not os.path.exists('setup.py'):
        print("❌ Error: setup.py not found. Make sure you're in the package root directory.")
        return False

    # Clean previous builds
    print("\n🧹 Cleaning previous builds")
    for dir_name in ['build', 'dist', '*.egg-info']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   Removed: {dir_name}")

    # Install/upgrade build tools
    if not run_command([sys.executable, '-m', 'pip', 'install', '--upgrade', 'build', 'twine'],
                       "Installing/upgrading build tools"):
        return False

    # Build the package
    if not run_command([sys.executable, '-m', 'build'],
                       "Building package"):
        return False

    # Check the built package
    if not run_command([sys.executable, '-m', 'twine', 'check', 'dist/*'],
                       "Checking package"):
        return False

    # Ask user if they want to upload
    print("\n📦 Package built successfully!")
    print("Files in dist/:")
    for file in os.listdir('dist'):
        print(f"   - {file}")

    upload_choice = input("\n🚀 Upload to TestPyPI? (y/n): ").strip().lower()

    if upload_choice in ['y', 'yes']:
        print("\n📝 Note: You'll need your TestPyPI API token")
        print("   Get it from: https://test.pypi.org/manage/account/token/")
        print("   Username: __token__")
        print("   Password: your-api-token")

        # Upload to TestPyPI
        if not run_command([sys.executable, '-m', 'twine', 'upload', '--repository', 'testpypi', 'dist/*'],
                           "Uploading to TestPyPI"):
            return False

        print("\n🎉 Package uploaded successfully!")
        print("\n📦 To install from TestPyPI:")
        print("   pip install -i https://test.pypi.org/simple/ enhanced-dcgan-research")
        print("\n🔗 View your package at:")
        print("   https://test.pypi.org/project/enhanced-dcgan-research/")

    else:
        print("\n📦 Package built but not uploaded")
        print("   To upload later: python -m twine upload --repository testpypi dist/*")

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    print("\n✅ Build process completed!")