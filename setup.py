#!/usr/bin/env python3
"""
Setup script for Enhanced DCGAN Research Framework
Production-Ready with Multi-Session Analytics and Complete Session Management
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Enhanced DCGAN Research Framework with Multi-Session Analytics and Complete Session Management"

# CORE DEPENDENCIES - All essential packages included in base install
CORE_REQUIREMENTS = [
    # Deep Learning Framework
    "torch>=2.0.0",
    "torchvision>=0.15.0",

    # Scientific Computing (Essential)
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",

    # Visualization (Essential)
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "Pillow>=8.0.0",

    # Progress & Monitoring (Essential)
    "tqdm>=4.60.0",
    "tensorboard>=2.8.0",

    # System Utilities (Essential for session management)
    "psutil>=5.8.0",

    # Data Science (Essential for analytics)
    "scikit-learn>=1.0.0",

    # Configuration (for YAML configs if needed)
    "pyyaml>=6.0",
]

# OPTIONAL DEPENDENCIES
EXTRAS_REQUIRE = {
    # Web interface support
    'web': [
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.11.5",
        "websocket-client>=1.8.0",
    ],

    # Development tools
    'dev': [
        "pytest>=6.0.0",
        "pytest-cov>=2.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "build>=0.7.0",
        "twine>=4.0.0",
    ],
}

# Complete installation with all features
EXTRAS_REQUIRE['all'] = (
        EXTRAS_REQUIRE['web'] +
        EXTRAS_REQUIRE['dev']
)

setup(
    name="enhanced-dcgan-research",
    version="0.1.4",
    author="Jahidul Arafat",
    author_email="jahidapon@gmail.com",
    description="Production-Ready Enhanced DCGAN Research Framework with Multi-Session Analytics and Complete Session Management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jahidul-arafat/gan-mnist-cifar",
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
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "enhanced-dcgan=enhanced_dcgan_research.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        'enhanced_dcgan_research': ['*.py', 'data/*'],
    },
    project_urls={
        "Bug Reports": "https://github.com/jahidul-arafat/gan-mnist-cifar/issues",
        "Source": "https://github.com/jahidul-arafat/gan-mnist-cifar",
        "Documentation": "https://github.com/jahidul-arafat/gan-mnist-cifar/blob/main/README.md",
    },
    keywords=["dcgan", "gan", "deep-learning", "pytorch", "academic-research", "multi-session", "analytics", "checkpointing"],
    zip_safe=False,
)