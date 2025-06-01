#!/usr/bin/env python3
"""
Setup script for Enhanced DCGAN Research Framework
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="enhanced-dcgan-research",
    version="0.1.0",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="Enhanced DCGAN Research Framework with Academic Reporting and Advanced Checkpointing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enhanced-dcgan-research",  # Replace with your repo
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        'tensorboard': ['tensorboard>=2.8.0'],
        'psutil': ['psutil>=5.8.0'],
        'dev': [
            'pytest>=6.0.0',
            'build>=0.7.0',
            'twine>=4.0.0',
        ],
    },
    entry_points={
        "console_scripts": [
            "enhanced-dcgan=enhanced_dcgan_research.fully_integrated_report_v04:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)