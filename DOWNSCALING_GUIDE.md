# Image Downscaling for Stereo Vision

This guide explains how to downscale images for stereo processing and properly adjust camera parameters.

## Quick Answer

**When you downscale images by factor `s` (e.g., 0.5 for half size):**

### What Changes:
- ✅ **Intrinsics (K matrix)**: Multiply `fx`, `fy`, `cx`, `cy` by scale factor `s`
- ❌ **Baseline**: Does NOT change (physical distance stays the same)

### What Happens to Point Cloud:
- ✅ **Automatically correct world coordinates** - No scaling needed!
- The scaled focal length in K already accounts for the image scaling

## Why This Works

The depth formula is:
```
depth = (focal_length × baseline) / disparity
```

When you downscale images by 0.5:
- Disparity values become 2× smaller (because features are 2× closer in pixels)
- Focal length becomes 2× smaller (from scaling K)
- These cancel out: `(0.5×f × b) / (0.5×d) = (f × b) / d` ✓

**Result**: Point cloud is in correct world coordinates automatically!

## Usage

### 1. Downscale Images

```bash
# Single image
python scripts/downscale_image.py ./assets/left_gazebo_image.png ./assets/right_gazebo_image.png ./assets/K_firefly_gazebo.txt --scale 0.6
```

### 1. Resulting Images and Camera Matrix

The x focal length and principal point (optical center) are scaled by the x scale, and the y focal length and principal point are scaled by the y scale.

```bash
Original size: 1440×1080
Original K matrix:
[[858.06269646   0.         720.        ]
 [  0.         858.06263208 540.        ]
 [  0.           0.           1.        ]]
Baseline: 0.06

Target scale: 0.6
Target size (before rounding): 864×648
Final size (divisible by 224): 896×672
  672 ÷ 224 = 3
  896 ÷ 224 = 4

Actual scales: w=0.622222, h=0.622222

Updated K matrix:
[[533.9056778    0.         448.        ]
 [  0.         533.90563774 336.        ]
 [  0.           0.           1.        ]]
Baseline (unchanged): 0.06
```

### 3. Use Scaled Parameters

For FoundationStereo ONNX export with half-size 540×720 images:

```bash
# Original: 1080x1440
python scripts/make_onnx.py \
    --height 1080 --width 1440 \
    --K_path assets/K_original.txt \
    ...

# Half size: 540×720
python scripts/make_onnx.py \
    --height 540 --width 720 \
    --K_path assets/K_half.txt \
    ...
```

## Example: Scaling Firefly Gazebo Camera

If your original K matrix is:
```
858.0    0.0   720.0
  0.0  858.0   540.0
  0.0    0.0     1.0
```

For half-size images (scale=0.5):
```
429.0    0.0   360.0
  0.0  429.0   270.0
  0.0    0.0     1.0
```

**Baseline stays the same** (e.g., 0.06 m)

## Point Cloud Reconstruction

No additional scaling needed! Use the scaled K matrix and original baseline:

```python
# Pseudocode for point cloud
for each pixel (u, v) with disparity d:
    # Use scaled focal length from K_scaled
    Z = (fx_scaled * baseline) / d
    X = (u - cx_scaled) * Z / fx_scaled
    Y = (v - cy_scaled) * Z / fy_scaled
    
# Points are already in correct world coordinates!
```

## Common Pitfalls

❌ **Don't** scale baseline  
❌ **Don't** multiply point cloud coordinates by scale factor  
✅ **Do** scale K matrix intrinsics  
✅ **Do** use scaled images with scaled K  

## Benefits of Downscaling

- 🚀 **4× faster** processing (half width × half height)
- 💾 **4× less memory** usage
- 📊 Same accuracy for many applications
- 🎯 Correct world-scale point clouds
