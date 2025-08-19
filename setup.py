#!/usr/bin/env python3
"""
Setup script for Maize Leaf Classification project
This script helps set up the environment and verify all dependencies.
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install packages from requirements.txt"""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def verify_packages():
    """Verify that all required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'timm', 'pandas', 'numpy', 
        'sklearn', 'matplotlib', 'seaborn', 'PIL', 'tqdm'
    ]
    
    print("🔍 Verifying package installations...")
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not found")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All packages verified successfully!")
        return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'Train.csv',
        'Test.csv', 
        'SampleSubmission.csv',
        'Images'
    ]
    
    print("📁 Checking data files...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            if file == 'Images':
                img_count = len(os.listdir(file)) if os.path.isdir(file) else 0
                print(f"✅ {file}/ - {img_count} images found")
            else:
                print(f"✅ {file}")
        else:
            print(f"❌ {file} - Not found")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("📥 Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/chrismundwa/somalia-hackerthon")
        return False
    else:
        print("\n✅ All data files found!")
        return True

def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU available: {gpu_name}")
            return True
        else:
            print("⚠️  No GPU available - training will use CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed - cannot check GPU")
        return False

def main():
    """Main setup function"""
    print("🌽 Maize Leaf Classification - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Verify packages
    if not verify_packages():
        print("\n🔧 Try running: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check data files
    data_ok = check_data_files()
    
    # Check GPU
    gpu_available = check_gpu()
    
    print("\n" + "=" * 50)
    if data_ok:
        print("🎉 Setup completed successfully!")
        print("🚀 You can now run: jupyter notebook FinalSubmission.ipynb")
    else:
        print("⚠️  Setup completed with warnings")
        print("📥 Please download the dataset before running the notebook")
    
    if not gpu_available:
        print("💡 Consider using Google Colab or Kaggle for GPU acceleration")

if __name__ == "__main__":
    main()
