# M2 Pro GPU Acceleration Checker

This project contains a script to check if your M2 Pro (or other Apple Silicon) Mac has proper GPU acceleration setup using PyTorch's MPS (Metal Performance Shaders) backend.

## What This Script Does

The `check_mps.py` script verifies:
- Whether MPS (Metal Performance Shaders) is available on your system
- If you can create tensors and perform operations on the MPS device
- Basic information about your PyTorch installation

## Requirements

- Python 3.7 or higher
- PyTorch 1.12 or higher (with MPS support)

## Installation

1. Make sure you have Python installed
2. Install PyTorch with MPS support:
   ```bash
   pip install torch torchvision torchaudio
   ```

## Usage

Run the script with:
```bash
python check_mps.py
```

## Expected Output

If MPS is available, you should see output similar to:
```
Checking MPS (Metal Performance Shaders) GPU acceleration...
==================================================
✅ MPS is AVAILABLE
   MPS version: 1.0
✅ Tensor creation and movement to MPS successful
   Tensor device: mps:0
✅ MPS device count: 1
==================================================
PyTorch Information:
   PyTorch version: 2.0.1
   CUDA available: False
   MPS available: True

Testing MPS operations:
✅ Matrix multiplication on MPS successful
   Result shape: torch.Size([1000, 1000])
   Result device: mps:0
```

If MPS is not available, you'll see a message indicating that your system doesn't support MPS or PyTorch wasn't compiled with MPS support.

## Troubleshooting

If you encounter issues:
1. Ensure you're running on an Apple Silicon Mac (M1, M2, M3, etc.)
2. Make sure you have the latest version of PyTorch installed
3. Verify that your Python environment is properly configured