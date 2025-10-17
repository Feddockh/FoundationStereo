# FoundationStereo Modifications

This document summarizes the modifications made to the FoundationStereo submodule for compatibility with our Python-based TensorRT installation.

## Changes Made

### 1. ONNX Export Configuration (`make_onnx.py`)
- **Change**: Updated ONNX opset version from 16 to 17
- **Reason**: Resolved LayerNorm compatibility issues during ONNX to TensorRT conversion

### 2. TensorRT Engine Builder (`make_tensorrt.py`)
- **Purpose**: Creates TensorRT engine files from ONNX models
- **Reason**: Replaces command-line `trtexec` tool which is unavailable in pip-based TensorRT installations (only included in Debian packages)

### 3. TensorRT Engine Import Path (`run_demo_tensorrt.py`)
- **Change**: Updated import path for `tensorrt_engine` module
- **Reason**: The `tensorrt_engine` package is not included in pip TensorRT distribution; copied directly from repository and updated import paths accordingly

## Additional Dependencies

The following Python packages were installed to support TensorRT inference:

```bash
pip install onnx==1.15.0
pip install onnxruntime-gpu==1.23.0
pip install tensorrt-cu12==10.13.3.9
pip install pycuda==2025.1.2
```

## Notes

- These modifications enable TensorRT inference without requiring the full NVIDIA TensorRT Debian package installation
- All changes maintain compatibility with the original model architecture and inference pipeline
- Please read this on installing TensorRT https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html
