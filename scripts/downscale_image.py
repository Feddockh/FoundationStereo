#!/usr/bin/env python3
"""
Downscale stereo images and update camera intrinsics with divisibility by 224 guarantee.
Usage: python downscale_image.py <left_image> <right_image> <K.txt> --scale 0.5
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

def load_K_and_baseline(k_path):
    """Load K matrix and baseline from file."""
    with open(k_path, 'r') as f:
        lines = f.readlines()
    
    # Parse K matrix (3x3 in first line or multiple lines)
    k_values = []
    baseline = None
    
    for line in lines:
        values = line.strip().split()
        if len(values) == 1:
            # Single value is likely the baseline
            baseline = float(values[0])
        else:
            k_values.extend([float(v) for v in values])
    
    # Reconstruct K matrix
    if len(k_values) >= 9:
        K = np.array(k_values[:9]).reshape(3, 3)
    else:
        raise ValueError(f"Invalid K matrix format in {k_path}")
    
    return K, baseline

def save_K_and_baseline(k_path, K, baseline):
    """Save K matrix and baseline to file."""
    with open(k_path, 'w') as f:
        # Write K as single line
        k_flat = K.flatten()
        f.write(' '.join([f"{v:.10f}" for v in k_flat]) + '\n')
        # Write baseline
        if baseline is not None:
            f.write(f"{baseline}\n")

def make_divisible_by_56(dimension):
    """Round dimension to nearest multiple of 56."""
    return round(dimension / 56) * 56

def make_divisible_by_224(dimension):
    """Round dimension to nearest multiple of 224."""
    return round(dimension / 224) * 224

def process_stereo_pair(left_path, right_path, k_path, output_left, output_right, output_k, 
                       target_scale=0.5, divisor=224):
    """
    Process stereo pair: downscale/crop/pad to ensure divisibility by 224.
    
    Args:
        left_path: Path to left image
        right_path: Path to right image
        k_path: Path to K matrix file
        output_left: Output path for left image
        output_right: Output path for right image
        output_k: Output path for K matrix
        target_scale: Target scale factor (approximate)
        divisor: Ensure dimensions divisible by this (default 224)
    """
    # Load images
    left_img = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
    right_img = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
    
    if left_img is None or right_img is None:
        raise ValueError("Could not read one or both images")
    
    if left_img.shape != right_img.shape:
        raise ValueError(f"Image size mismatch: {left_img.shape} vs {right_img.shape}")
    
    # Load K and baseline
    K, baseline = load_K_and_baseline(k_path)
    
    original_h, original_w = left_img.shape[:2]
    print(f"Original size: {original_w}×{original_h}")
    print(f"Original K matrix:")
    print(K)
    print(f"Baseline: {baseline}")
    print()
    
    # Calculate target dimensions
    target_w = int(original_w * target_scale)
    target_h = int(original_h * target_scale)
    
    # Round to nearest multiple of divisor
    final_w = make_divisible_by_224(target_w)
    final_h = make_divisible_by_224(target_h)
    
    print(f"Target scale: {target_scale}")
    print(f"Target size (before rounding): {target_w}×{target_h}")
    print(f"Final size (divisible by {divisor}): {final_w}×{final_h}")
    print(f"  {final_h} ÷ {divisor} = {final_h // divisor}")
    print(f"  {final_w} ÷ {divisor} = {final_w // divisor}")
    print()
    
    # Calculate actual scales
    actual_scale_w = final_w / original_w
    actual_scale_h = final_h / original_h
    
    print(f"Actual scales: w={actual_scale_w:.6f}, h={actual_scale_h:.6f}")
    
    # Resize images
    left_resized = cv2.resize(left_img, (final_w, final_h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_img, (final_w, final_h), interpolation=cv2.INTER_AREA)
    
    # Update K matrix
    K_new = K.copy()
    K_new[0, 0] *= actual_scale_w  # fx
    K_new[1, 1] *= actual_scale_h  # fy
    K_new[0, 2] *= actual_scale_w  # cx
    K_new[1, 2] *= actual_scale_h  # cy
    
    print()
    print(f"Updated K matrix:")
    print(K_new)
    print(f"Baseline (unchanged): {baseline}")
    print()
    
    # Save outputs
    output_left.parent.mkdir(parents=True, exist_ok=True)
    output_right.parent.mkdir(parents=True, exist_ok=True)
    output_k.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_left), left_resized)
    cv2.imwrite(str(output_right), right_resized)
    save_K_and_baseline(output_k, K_new, baseline)
    
    print(f"✓ Saved left image: {output_left}")
    print(f"✓ Saved right image: {output_right}")
    print(f"✓ Saved K matrix: {output_k}")
    
    return left_resized, right_resized, K_new, baseline

def main():
    parser = argparse.ArgumentParser(
        description="Downscale stereo pair and update camera intrinsics (ensures divisibility by 56)"
    )
    parser.add_argument("left", type=str, help="Left image path")
    parser.add_argument("right", type=str, help="Right image path")
    parser.add_argument("k_matrix", type=str, help="K matrix file path (with baseline)")
    parser.add_argument("--output_left", type=str, default=None,
                       help="Output left image path (default: <left>_scaled.png)")
    parser.add_argument("--output_right", type=str, default=None,
                       help="Output right image path (default: <right>_scaled.png)")
    parser.add_argument("--output_k", type=str, default=None,
                       help="Output K matrix path (default: <k>_scaled.txt)")
    parser.add_argument("--scale", type=float, default=0.5, 
                       help="Target scale factor (default: 0.5)")
    parser.add_argument("--divisor", type=int, default=224,
                       help="Ensure dimensions divisible by this value (default: 224)")
    
    args = parser.parse_args()
    
    # Setup paths
    left_path = Path(args.left)
    right_path = Path(args.right)
    k_path = Path(args.k_matrix)
    
    if not left_path.exists():
        print(f"Error: Left image not found: {left_path}")
        return
    if not right_path.exists():
        print(f"Error: Right image not found: {right_path}")
        return
    if not k_path.exists():
        print(f"Error: K matrix not found: {k_path}")
        return
    
    # Setup output paths
    if args.output_left:
        output_left = Path(args.output_left)
    else:
        output_left = left_path.parent / f"{left_path.stem}_scaled{left_path.suffix}"
    
    if args.output_right:
        output_right = Path(args.output_right)
    else:
        output_right = right_path.parent / f"{right_path.stem}_scaled{right_path.suffix}"
    
    if args.output_k:
        output_k = Path(args.output_k)
    else:
        output_k = k_path.parent / f"{k_path.stem}_scaled{k_path.suffix}"
    
    # Process
    process_stereo_pair(
        left_path, right_path, k_path,
        output_left, output_right, output_k,
        target_scale=args.scale,
        divisor=args.divisor
    )

if __name__ == "__main__":
    main()
