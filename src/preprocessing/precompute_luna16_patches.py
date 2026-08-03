

import argparse
import gc
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch


def build_seriesuid_to_path_map(luna16_root):
    import glob
    mhd_paths = glob.glob(os.path.join(luna16_root, "subset*", "*.mhd"))
    if not mhd_paths:
        raise FileNotFoundError(f"No .mhd files found under {luna16_root}/subset*/")
    return {os.path.splitext(os.path.basename(p))[0]: p for p in mhd_paths}


def world_to_voxel(world_coord, origin, spacing):
    return np.absolute(world_coord - origin) / spacing


def load_candidates(luna16_root, subsets, max_negatives, seriesuid_to_path):
    candidates = pd.read_csv(os.path.join(luna16_root, "candidates_V2.csv"))

    if subsets is not None:
        valid_uids = {
            uid for uid, path in seriesuid_to_path.items()
            if any(f"subset{s}" in path for s in subsets)
        }
        candidates = candidates[candidates["seriesuid"].isin(valid_uids)]

    candidates = candidates[candidates["seriesuid"].isin(seriesuid_to_path.keys())]

    positives = candidates[candidates["class"] == 1]
    negatives = candidates[candidates["class"] == 0].sample(
        n=min(max_negatives, len(candidates[candidates["class"] == 0])),
        random_state=42,
    )
    candidates = pd.concat([positives, negatives]).reset_index(drop=True)

    # Sort by scan so extraction only needs one volume in memory at a time
    return candidates.sort_values("seriesuid").reset_index(drop=True)


def extract_2d_patch(slice_2d, voxel_y, voxel_x, patch_size=32):
    half = patch_size // 2
    y_min, y_max = voxel_y - half, voxel_y + half
    x_min, x_max = voxel_x - half, voxel_x + half

    pad_y_before, pad_x_before = max(0, -y_min), max(0, -x_min)
    pad_y_after = max(0, y_max - slice_2d.shape[0])
    pad_x_after = max(0, x_max - slice_2d.shape[1])

    img = slice_2d
    if pad_y_before or pad_x_before or pad_y_after or pad_x_after:
        img = np.pad(
            slice_2d,
            ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
            mode="constant", constant_values=-1000,
        )
        y_min += pad_y_before; y_max += pad_y_before
        x_min += pad_x_before; x_max += pad_x_before

    patch = img[y_min:y_max, x_min:x_max].astype(np.float32)
    patch = np.clip(patch, -1000, 400)
    return (patch + 1000) / 1400


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--luna16_root", required=True)
    parser.add_argument("--subsets", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max_negatives", type=int, default=600)
    parser.add_argument("--output", default="precomputed_patches.pt")
    args = parser.parse_args()

    seriesuid_to_path = build_seriesuid_to_path_map(args.luna16_root)
    candidates = load_candidates(args.luna16_root, args.subsets, args.max_negatives,
                                  seriesuid_to_path)
    print(f"Total candidates: {len(candidates)}")

    all_patches, all_labels = [], []
    unique_uids = candidates["seriesuid"].unique()
    print(f"Processing {len(unique_uids)} unique scans...")

    for scan_idx, seriesuid in enumerate(unique_uids):
        itk_img = sitk.ReadImage(seriesuid_to_path[seriesuid])
        volume = sitk.GetArrayFromImage(itk_img)
        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())

        scan_candidates = candidates[candidates["seriesuid"] == seriesuid]
        for _, row in scan_candidates.iterrows():
            world_coord = np.array([row["coordX"], row["coordY"], row["coordZ"]])
            voxel_coord = world_to_voxel(world_coord, origin, spacing)
            voxel_x, voxel_y, voxel_z = voxel_coord.astype(int)
            z = np.clip(voxel_z, 0, volume.shape[0] - 1)

            patch = extract_2d_patch(volume[z], voxel_y, voxel_x)
            all_patches.append(torch.from_numpy(patch.copy()).unsqueeze(0))
            all_labels.append(torch.tensor(float(row["class"]), dtype=torch.float32))

        # Explicit cleanup — this is the fix for the memory leak
        del itk_img, volume
        gc.collect()

        if scan_idx % 50 == 0:
            print(f"  scan {scan_idx}/{len(unique_uids)}")

    all_patches = torch.stack(all_patches)
    all_labels = torch.stack(all_labels)
    print(f"Final shape: {all_patches.shape}")

    torch.save({"patches": all_patches, "labels": all_labels}, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
