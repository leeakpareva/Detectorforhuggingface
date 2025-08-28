#!/usr/bin/env python3
"""
Dependency Checker for NAVADA AI Computer Vision Application
This script verifies that all required packages are properly installed.
"""

import sys
from importlib import import_module

# List of required packages
REQUIRED_PACKAGES = [
    ('gradio', 'Gradio web framework'),
    ('ultralytics', 'YOLOv8 object detection'),
    ('openai', 'OpenAI API client'),
    ('cv2', 'OpenCV computer vision'),
    ('PIL', 'Pillow image processing'),
    ('numpy', 'NumPy numerical computing'),
    ('torch', 'PyTorch machine learning'),
    ('torchvision', 'PyTorch computer vision'),
    ('plotly.graph_objects', 'Plotly interactive charts'),
    ('requests', 'HTTP requests library'),
]

def check_package(package_name, description):
    """Check if a package can be imported"""
    try:
        import_module(package_name)
        print(f"✅ {package_name:<25} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:<25} - {description} | Error: {str(e)}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_gpu_support():
    """Check GPU/CUDA support"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"✅ CUDA GPU Support - {gpu_count} device(s) | Primary: {gpu_name}")
            return True
        else:
            print("⚠️  CUDA GPU Support - Not available (CPU-only mode)")
            return False
    except ImportError:
        print("❌ CUDA GPU Support - PyTorch not installed")
        return False

def main():
    """Main dependency check function"""
    print("🔍 NAVADA Dependency Checker")
    print("=" * 60)
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Check packages
    print("📦 Package Dependencies:")
    print("-" * 60)
    
    failed_packages = []
    for package, description in REQUIRED_PACKAGES:
        if not check_package(package, description):
            failed_packages.append(package)
    
    print()
    
    # Check GPU support
    print("🔧 Hardware Support:")
    print("-" * 60)
    check_gpu_support()
    
    print()
    print("=" * 60)
    
    # Summary
    if not python_ok:
        print("❌ CRITICAL: Python version incompatible")
        print("   Please install Python 3.8 or higher")
        return False
    
    if failed_packages:
        print(f"❌ MISSING PACKAGES: {len(failed_packages)} package(s) not found")
        print("\n📋 To install missing packages:")
        print("   pip install -r requirements.txt")
        print("\n🔧 For individual packages:")
        for pkg in failed_packages:
            print(f"   pip install {pkg}")
        return False
    else:
        print("✅ ALL DEPENDENCIES SATISFIED")
        print("🚀 Your system is ready to run NAVADA!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)