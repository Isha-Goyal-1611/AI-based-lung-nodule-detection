"""
src/preprocessing/precompute_lung_masks.py

Extracts (CT slice, lung mask slice) pairs for training MiniUNet, using
LUNA16's seg-lungs-LUNA16.zip ground-truth segmentation masks.

The masks are stored as .mhd/.raw files matching the same seriesuid as
the scans, in a flat folder (not split into subset0/1/2/... like the
scans are). Each mask volume has the same shape as its corresponding
scan volume, with lung regions labeled (typically 3=right lung, 4=left
lung, 5=trachea — treat any value > 0 as "lung" for a binary mask).

Usage:
    python precompute_lung_masks.py --luna16_root /path/to/luna16 \
        --masks_dir /path/to/seg-lungs-luna16 \
        --subsets 0 1 2 --slices_per_scan 5 --image_size 128 \
        --output lung_mask_patches.pt
"""

import argparse
import glob
import gc
import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def build_seriesuid_to_path_map(luna16_root):
    mhd_paths = glob.glob(os.path.join(luna16_root, "subset*", "*.mhd"))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in mhd_paths}


def build_mask_map(masks_dir):
    mhd_paths = glob.glob(os.path.join(masks_dir, "*.mhd"))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in mhd_paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--luna16_root", required=True)
    parser.add_argument("--masks_dir", required=True,
                         help="Folder containing extracted seg-lungs-LUNA16 .mhd/.raw files")
    parser.add_argument("--subsets", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--slices_per_scan", type=int, default=5,
                         help="How many slices (with visible lung) to sample per scan")
    parser.add_argument("--image_size", type=int, default=128,
                         help="Resize slices to this size (must be divisible by 4 for MiniUNet)")
    parser.add_argument("--output", default="lung_mask_patches.pt")
    args = parser.parse_args()

    scan_paths = build_seriesuid_to_path_map(args.luna16_root)
    mask_paths = build_mask_map(args.masks_dir)

    # Only use scans that have both a CT volume AND a matching mask
    valid_uids = set()
    for uid, path in scan_paths.items():
        if any(f"subset{s}" in path for s in args.subsets) and uid in mask_paths:
            valid_uids.add(uid)
    print(f"Found {len(valid_uids)} scans with matching masks in selected subsets")

    all_images, all_masks = [], []

    for i, uid in enumerate(valid_uids):
        ct_img = sitk.ReadImage(scan_paths[uid])
        ct_volume = sitk.GetArrayFromImage(ct_img)  # (z, y, x)

        mask_img = sitk.ReadImage(mask_paths[uid])
        mask_volume = sitk.GetArrayFromImage(mask_img)  # (z, y, x), values 0/3/4/5

        # Binary mask: anything > 0 counts as "lung" (includes trachea, both lungs)
        binary_mask = (mask_volume > 0).astype(np.float32)

        # Pick slices that actually contain some lung, spaced through the volume
        lung_slice_indices = [z for z in range(binary_mask.shape[0])
                               if binary_mask[z].sum() > 100]
        if not lung_slice_indices:
            del ct_img, ct_volume, mask_img, mask_volume
            gc.collect()
            continue

        step = max(1, len(lung_slice_indices) // args.slices_per_scan)
        chosen = lung_slice_indices[::step][:args.slices_per_scan]

        for z in chosen:
            ct_slice = ct_volume[z].astype(np.float32)
            ct_slice = np.clip(ct_slice, -1000, 400)
            ct_slice = (ct_slice + 1000) / 1400

            mask_slice = binary_mask[z]

            ct_tensor = torch.from_numpy(ct_slice).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).unsqueeze(0)

            ct_resized = F.interpolate(ct_tensor, size=(args.image_size, args.image_size),
                                        mode="bilinear", align_corners=False)
            mask_resized = F.interpolate(mask_tensor, size=(args.image_size, args.image_size),
                                          mode="nearest")

            all_images.append(ct_resized.squeeze(0))
            all_masks.append((mask_resized.squeeze(0) > 0.5).float())

        del ct_img, ct_volume, mask_img, mask_volume
        gc.collect()

        if i % 20 == 0:
            print(f"  scan {i}/{len(valid_uids)}, {len(all_images)} slice pairs so far")

    all_images = torch.stack(all_images)
    all_masks = torch.stack(all_masks)
    print(f"Final: images {all_images.shape}, masks {all_masks.shape}")

    torch.save({"images": all_images, "masks": all_masks}, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()