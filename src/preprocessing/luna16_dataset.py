"""
Day 1-2: LUNA16 Dataset Loader
──────────────────────────────
Replaces the fake tensors in train.py with real labeled nodule/non-nodule patches
pulled from LUNA16.
"""

import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

LUNA16_ROOT = "C:/lung_data/luna16"
PATCH_SIZE = 32  # 2D patch: PATCH_SIZE x PATCH_SIZE, centered on candidate


def build_seriesuid_to_path_map(luna16_root=LUNA16_ROOT):
    """
    LUNA16 scans are split across subset0..subset9 folders.
    candidates_V2.csv only gives you a seriesuid, not which subset it's in,
    so we need to build a lookup table first.
    """
    mhd_paths = glob.glob(os.path.join(luna16_root, "subset*", "*.mhd"))
    if not mhd_paths:
        raise FileNotFoundError(
            f"No .mhd files found under {luna16_root}/subset*/. "
            "Check LUNA16_ROOT and that subsets were extracted."
        )
    seriesuid_to_path = {
        os.path.splitext(os.path.basename(p))[0]: p for p in mhd_paths
    }
    return seriesuid_to_path


def world_to_voxel(world_coord, origin, spacing):
    """
    candidates_V2.csv / annotations.csv give coordinates in world (mm) space.
    The image array is indexed in voxel space. This converts between them.
    """
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord


class LUNA16PatchDataset(Dataset):
    """
    Returns (patch, label) pairs.
    patch: torch.FloatTensor, shape (1, PATCH_SIZE, PATCH_SIZE)
    label: 1.0 if real nodule (class==1), else 0.0
    """

    def __init__(self, luna16_root=LUNA16_ROOT, subsets=None, patch_size=PATCH_SIZE,
                 max_candidates=None):
        self.luna16_root = luna16_root
        self.patch_size = patch_size
        self.seriesuid_to_path = build_seriesuid_to_path_map(luna16_root)

        candidates_path = os.path.join(luna16_root, "candidates_V2.csv")
        if not os.path.exists(candidates_path):
            raise FileNotFoundError(f"candidates_V2.csv not found at {candidates_path}")
        candidates = pd.read_csv(candidates_path)

        if subsets is not None:
            valid_uids = {
                uid for uid, path in self.seriesuid_to_path.items()
                if any(f"subset{s}" in path for s in subsets)
            }
            candidates = candidates[candidates["seriesuid"].isin(valid_uids)]

        candidates = candidates[candidates["seriesuid"].isin(self.seriesuid_to_path.keys())]

        if max_candidates is not None:
            positives = candidates[candidates["class"] == 1]
            negatives = candidates[candidates["class"] == 0].sample(
                n=min(max_candidates, len(candidates[candidates["class"] == 0])),
                random_state=42,
            )
            candidates = pd.concat([positives, negatives]).reset_index(drop=True)

        self.candidates = candidates.reset_index(drop=True)
        self._volume_cache = {}

    def __len__(self):
        return len(self.candidates)

    def _load_volume(self, seriesuid):
        if seriesuid not in self._volume_cache:
            path = self.seriesuid_to_path[seriesuid]
            itk_img = sitk.ReadImage(path)
            volume = sitk.GetArrayFromImage(itk_img)
            origin = np.array(itk_img.GetOrigin())
            spacing = np.array(itk_img.GetSpacing())
            self._volume_cache[seriesuid] = (volume, origin, spacing)
            if len(self._volume_cache) > 5:
                oldest_key = next(iter(self._volume_cache))
                if oldest_key != seriesuid:
                    del self._volume_cache[oldest_key]
        return self._volume_cache[seriesuid]

    def __getitem__(self, idx):
        row = self.candidates.iloc[idx]
        seriesuid = row["seriesuid"]
        world_coord = np.array([row["coordX"], row["coordY"], row["coordZ"]])
        label = float(row["class"])

        volume, origin, spacing = self._load_volume(seriesuid)
        voxel_coord = world_to_voxel(world_coord, origin, spacing)
        voxel_x, voxel_y, voxel_z = voxel_coord.astype(int)

        z = np.clip(voxel_z, 0, volume.shape[0] - 1)
        slice_2d = volume[z]

        half = self.patch_size // 2
        y_min, y_max = voxel_y - half, voxel_y + half
        x_min, x_max = voxel_x - half, voxel_x + half

        pad_y_before = max(0, -y_min)
        pad_x_before = max(0, -x_min)
        pad_y_after = max(0, y_max - slice_2d.shape[0])
        pad_x_after = max(0, x_max - slice_2d.shape[1])

        if pad_y_before or pad_x_before or pad_y_after or pad_x_after:
            slice_2d = np.pad(
                slice_2d,
                ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
                mode="constant",
                constant_values=-1000,
            )
            y_min += pad_y_before
            y_max += pad_y_before
            x_min += pad_x_before
            x_max += pad_x_before

        patch = slice_2d[y_min:y_max, x_min:x_max].astype(np.float32)
        patch = np.clip(patch, -1000, 400)
        patch = (patch + 1000) / 1400

        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return patch_tensor, label_tensor


if __name__ == "__main__":
    dataset = LUNA16PatchDataset(subsets=[0], max_candidates=200)
    print(f"Dataset size: {len(dataset)}")
    print(f"Positives:    {(dataset.candidates['class'] == 1).sum()}")
    print(f"Negatives:    {(dataset.candidates['class'] == 0).sum()}")

    patch, label = dataset[0]
    print(f"Patch shape:  {patch.shape}")
    print(f"Patch dtype:  {patch.dtype}")
    print(f"Label:        {label.item()}")
    print(f"Patch range:  [{patch.min().item():.3f}, {patch.max().item():.3f}]")