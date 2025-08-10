#!/usr/bin/env python3
"""
Setup script for MRI Brain Analysis System
=========================================

This script installs the brain analysis system as a Python package.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements_file = this_directory / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="mri-brain-analyzer",
    version="1.0.0",
    author="TendrilAI Team",
    author_email="contact@tendrilai.com",
    description="Advanced MRI brain analysis system with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TendrilAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "web": [
            "streamlit>=1.25.0",
            "gradio>=3.35.0",
        ],
        "full": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "tensorflow>=2.13.0",
            "keras>=2.13.0",
            "monai>=1.2.0",
            "nibabel>=5.1.0",
            "SimpleITK>=2.2.0",
            "pydicom>=2.4.0",
            "plotly>=5.15.0",
            "pyvista>=0.40.0",
            "nilearn>=0.10.0",
            "dipy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain-analyzer=scripts.download_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "brain_analyzer": ["*.json", "*.txt"],
    },
    keywords=[
        "mri", "brain", "analysis", "deep-learning", "medical-imaging",
        "segmentation", "abnormality-detection", "neuroscience", "radiology"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/TendrilAI/issues",
        "Source": "https://github.com/yourusername/TendrilAI",
        "Documentation": "https://github.com/yourusername/TendrilAI/blob/main/README.md",
    },
) 