#!/usr/bin/env python3
"""
Script to check M2 Pro GPU acceleration with torch.backends.mps
"""

import torch

def check_mps_acceleration():
    """Check if MPS (Metal Performance Shaders) is available for GPU acceleration"""
    
    print("Checking MPS (Metal Performance Shaders) GPU acceleration...")
    print("=" * 50)
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("✅ MPS is AVAILABLE")
        print(f"   MPS version: {torch.backends.mps.version()}")
        
        # Create a tensor and move it to MPS
        try:
            x = torch.randn(3, 3).to('mps')
            print("✅ Tensor creation and movement to MPS successful")
            print(f"   Tensor device: {x.device}")
        except Exception as e:
            print(f"❌ Error creating tensor on MPS: {e}")
            
        # Check MPS device count
        try:
            device_count = torch.backends.mps.device_count()
            print(f"✅ MPS device count: {device_count}")
        except Exception as e:
            print(f"❌ Error checking MPS device count: {e}")
            
    else:
        print("❌ MPS is NOT AVAILABLE")
        print("   This might be because:")
        print("   - You're not on an Apple Silicon Mac (M1, M2, M3, etc.)")
        print("   - PyTorch was not compiled with MPS support")
        print("   - Your system doesn't have the required Metal framework")
    
    print("=" * 50)
    
    # General PyTorch information
    print("PyTorch Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    
    # If MPS is available, test some operations
    if torch.backends.mps.is_available():
        print("\nTesting MPS operations:")
        try:
            # Create some tensors
            a = torch.randn(1000, 1000).to('mps')
            b = torch.randn(1000, 1000).to('mps')
            
            # Perform matrix multiplication
            c = torch.mm(a, b)
            print("✅ Matrix multiplication on MPS successful")
            print(f"   Result shape: {c.shape}")
            print(f"   Result device: {c.device}")
            
        except Exception as e:
            print(f"❌ Error in MPS operations: {e}")

if __name__ == "__main__":
    check_mps_acceleration()