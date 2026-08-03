"""
src/preprocessing/lidc_extraction.py

Downloads LIDC-IDRI patients via TCIA, reorganizes for pylidc, clusters
multi-radiologist annotations per nodule, and extracts 3D patches with
consensus malignancy labels + clinical-style features (diameter, subtlety,
texture).

IMPORTANT BUG FIX NOTE: earlier versions of this extraction swapped the
centroid axis order (used cz,cy,cx instead of cx,cy,cz), which silently
extracted empty/padding patches for most nodules. pylidc's annotation
centroid already matches scan.to_volume()'s array axis order directly —
do NOT reorder it. Verified by comparing ann.centroid against ann.bbox()
on a known scan; they line up in the same order.

Requires (install once per session):
    pip install SimpleITK tcia_utils pylidc

Also requires these compatibility patches for modern Python/numpy, since
pylidc is an older library:
    import configparser; configparser.SafeConfigParser = configparser.ConfigParser
    import numpy as np; np.int = int; np.float = float; np.bool = bool

Usage:
    python lidc_extraction.py --n_patients 75 --data_dir /content/data/lidc \
        --output malignancy_data.pt
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch


def apply_compatibility_patches():
    import configparser
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "bool"):
        np.bool = bool


def download_patients(n_patients, data_dir):
    from tcia_utils import nbia
    series_data = nbia.getSeries(collection="LIDC-IDRI", modality="CT")
    df = pd.DataFrame(series_data)
    subset_df = df.drop_duplicates(subset="PatientID").head(n_patients)
    print(f"Selected {len(subset_df)} patients")

    series_uids = subset_df["SeriesInstanceUID"].tolist()
    nbia.downloadSeries(series_uids, input_type="list", path=data_dir)
    print("Download complete")
    return subset_df


def reorganize_for_pylidc(subset_df, data_dir):
    """pylidc expects data_dir/PatientID/... ; tcia_utils gives data_dir/SeriesUID/..."""
    reorganized = 0
    for _, row in subset_df.iterrows():
        patient_id, series_uid = row["PatientID"], row["SeriesInstanceUID"]
        old_path = os.path.join(data_dir, series_uid)
        new_folder = os.path.join(data_dir, patient_id)
        if os.path.exists(old_path) and not os.path.exists(new_folder):
            os.makedirs(new_folder, exist_ok=True)
            shutil.move(old_path, os.path.join(new_folder, series_uid))
            reorganized += 1
    print(f"Reorganized {reorganized} patient folders")


def write_pylidc_config(data_dir):
    config_content = f"[dicom]\npath = {data_dir}\nwarn = True\n"
    with open(os.path.expanduser("~/.pylidcrc"), "w") as f:
        f.write(config_content)


def extract_3d_patch(volume, center_voxel, patch_size=32):
    """
    center_voxel comes from pylidc's ann.centroid, which already matches
    volume's array axis order directly (x, y, z) -> (axis0, axis1, axis2).
    Do NOT reorder these values.
    """
    half = patch_size // 2
    cx, cy, cz = center_voxel

    x_min, x_max = int(cx - half), int(cx + half)
    y_min, y_max = int(cy - half), int(cy + half)
    z_min, z_max = int(cz - half), int(cz + half)

    pad_x_b, pad_y_b, pad_z_b = max(0, -x_min), max(0, -y_min), max(0, -z_min)
    pad_x_a = max(0, x_max - volume.shape[0])
    pad_y_a = max(0, y_max - volume.shape[1])
    pad_z_a = max(0, z_max - volume.shape[2])

    vol = volume
    if any([pad_x_b, pad_y_b, pad_z_b, pad_x_a, pad_y_a, pad_z_a]):
        vol = np.pad(volume, ((pad_x_b, pad_x_a), (pad_y_b, pad_y_a), (pad_z_b, pad_z_a)),
                     mode="constant", constant_values=-1000)
        x_min += pad_x_b; x_max += pad_x_b
        y_min += pad_y_b; y_max += pad_y_b
        z_min += pad_z_b; z_max += pad_z_b

    patch = vol[x_min:x_max, y_min:y_max, z_min:z_max].astype(np.float32)
    patch = np.clip(patch, -1000, 400)
    return (patch + 1000) / 1400


def extract_all_nodules(subset_df, data_dir):
    import pylidc as pl

    all_patches, all_labels, all_clinical = [], [], []
    patient_ids = subset_df["PatientID"].unique()
    print(f"Extracting nodules from {len(patient_ids)} patients...")

    for i, pid in enumerate(patient_ids):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        if scan is None:
            continue
        try:
            volume = scan.to_volume()
        except Exception as e:
            print(f"  {pid}: failed to load ({e}), skipping")
            continue

        for cluster in scan.cluster_annotations():
            malignancies = [ann.malignancy for ann in cluster]
            diameters = [ann.diameter for ann in cluster]
            subtleties = [ann.subtlety for ann in cluster]
            textures = [ann.texture for ann in cluster]

            centroid = cluster[0].centroid
            patch = extract_3d_patch(volume, centroid)
            if patch.shape != (32, 32, 32):
                continue

            all_patches.append(torch.from_numpy(patch).unsqueeze(0))
            all_labels.append(sum(malignancies) / len(malignancies))
            all_clinical.append([
                sum(diameters) / len(diameters),
                sum(subtleties) / len(subtleties),
                sum(textures) / len(textures),
            ])

        if i % 10 == 0:
            print(f"  {i}/{len(patient_ids)} patients, {len(all_patches)} nodules so far")

    patches_tensor = torch.stack(all_patches)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    clinical_tensor = torch.tensor(all_clinical, dtype=torch.float32)

    n_flat = sum((p.max() - p.min()).item() < 0.01 for p in patches_tensor)
    print(f"Total: {len(patches_tensor)} nodules. Flat/broken patches: {n_flat} (should be 0)")

    return patches_tensor, labels_tensor, clinical_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patients", type=int, default=75)
    parser.add_argument("--data_dir", default="/content/data/lidc")
    parser.add_argument("--output", default="malignancy_data.pt")
    parser.add_argument("--skip_download", action="store_true",
                         help="Use if data_dir is already populated")
    args = parser.parse_args()

    apply_compatibility_patches()

    if args.skip_download:
        from tcia_utils import nbia
        series_data = nbia.getSeries(collection="LIDC-IDRI", modality="CT")
        subset_df = pd.DataFrame(series_data).drop_duplicates(
            subset="PatientID").head(args.n_patients)
    else:
        subset_df = download_patients(args.n_patients, args.data_dir)
        reorganize_for_pylidc(subset_df, args.data_dir)

    write_pylidc_config(args.data_dir)

    patches, labels, clinical = extract_all_nodules(subset_df, args.data_dir)
    torch.save({"patches": patches, "labels": labels, "clinical": clinical}, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
