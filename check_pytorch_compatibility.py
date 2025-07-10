#!/usr/bin/env python3
"""
Check PyTorch version compatibility for SUPIR-Demo
"""
import sys

print("Checking PyTorch compatibility...")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # Check minimum version requirement
    torch_version = torch.__version__.split('+')[0]
    major, minor, patch = map(int, torch_version.split('.')[:3])
    
    if major >= 2 and minor >= 1:
        print("✓ PyTorch version meets minimum requirements (2.1.0+)")
    else:
        print("✗ PyTorch version too old, need at least 2.1.0")
        sys.exit(1)
        
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

print("\nCompatibility check passed!")