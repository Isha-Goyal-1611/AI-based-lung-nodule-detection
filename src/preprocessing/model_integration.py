"""
src/preprocessing/model_integration.py

Wires trained models (nodule classifiers, lung segmentation U-Net) into
full_pipeline.py, replacing/augmenting the classical heuristics.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure

from models.cnn_2d import SimpleCNN

PATCH_SIZE = 32
CHECKPOINT_PATH = "checkpoints/best_model_gpu.pt"

_model_cache = {"model": None}
_unet_model_cache = {"model": None}
_resnet_model_cache = {"model": None}


def get_trained_model():
    """
    Load the trained SimpleCNN nodule classifier once and cache it.
    """
    if _model_cache["model"] is None:
        model = SimpleCNN()
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _model_cache["model"] = model
        print(f"      [model] Loaded checkpoint from epoch {checkpoint['epoch']}, "
              f"val_loss {checkpoint['val_loss']:.4f}")
    return _model_cache["model"]


def extract_patch(hu_image, center_y, center_x, patch_size=PATCH_SIZE):
    """
    Same patch extraction + normalization convention as luna16_dataset.py's
    __getitem__, so the model sees consistent input at inference time.
    """
    half = patch_size // 2
    y_min, y_max = int(center_y - half), int(center_y + half)
    x_min, x_max = int(center_x - half), int(center_x + half)

    pad_y_before = max(0, -y_min)
    pad_x_before = max(0, -x_min)
    pad_y_after = max(0, y_max - hu_image.shape[0])
    pad_x_after = max(0, x_max - hu_image.shape[1])

    image = hu_image
    if pad_y_before or pad_x_before or pad_y_after or pad_x_after:
        image = np.pad(
            hu_image,
            ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
            mode="constant",
            constant_values=-1000,
        )
        y_min += pad_y_before
        y_max += pad_y_before
        x_min += pad_x_before
        x_max += pad_x_before

    patch = image[y_min:y_max, x_min:x_max].astype(np.float32)
    patch = np.clip(patch, -1000, 400)
    patch = (patch + 1000) / 1400
    return patch


def find_candidates_with_model(hu_image, lung_mask, threshold=0.5):
    """
    Finds candidates using the classical 'holes in mask' method, scored by
    the trained SimpleCNN. NOTE: this hole-based method implicitly relies
    on the classical segment_lungs()'s coincidental gaps around dense
    tissue — it does NOT work correctly with U-Net's cleaner, solid masks.
    See find_candidates_density_based() for a segmentation-agnostic version.
    """
    model = get_trained_model()

    filled_lung = ndimage.binary_fill_holes(lung_mask)
    candidate_mask = filled_lung.astype(int) - lung_mask.astype(int)
    labeled = measure.label(candidate_mask)
    regions = measure.regionprops(labeled, intensity_image=hu_image)

    candidates = []
    for region in regions:
        if region.area < 5 or region.area > 1000:
            continue

        y, x = region.centroid
        patch = extract_patch(hu_image, y, x)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            confidence = model(patch_tensor).item()

        if confidence < threshold:
            continue

        candidates.append({
            'x': x, 'y': y,
            'area': region.area,
            'mean_intensity': region.mean_intensity,
            'confidence': confidence,
        })

    return pd.DataFrame(candidates,
                         columns=['x', 'y', 'area', 'mean_intensity', 'confidence']) \
        if candidates else pd.DataFrame(
            columns=['x', 'y', 'area', 'mean_intensity', 'confidence'])


def get_unet_model():
    """
    Load the trained U-Net segmentation model once and cache it.
    """
    if _unet_model_cache["model"] is None:
        from models.unet import MiniUNet
        model = MiniUNet()
        checkpoint = torch.load("checkpoints/unet_model_40ep.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _unet_model_cache["model"] = model
        print(f"      [unet] Loaded checkpoint from epoch {checkpoint['epoch']}, "
              f"dice {checkpoint['val_dice']:.4f}")
    return _unet_model_cache["model"]


def segment_lungs_with_unet(hu_image, warnings):
    """
    U-Net-based alternative to the classical threshold segment_lungs() in
    full_pipeline.py. Same input/output contract: takes hu_image, returns
    a binary lung mask at the ORIGINAL image resolution.

    IMPORTANT: this produces a CLEANER, more solid lung mask than the
    classical method. Because of that, find_candidates_with_model()'s
    hole-based candidate detection will NOT work correctly with this
    mask — use find_candidates_density_based() instead when segmenting
    with U-Net. See that function's docstring for why.
    """
    model = get_unet_model()
    original_shape = hu_image.shape

    normalized = np.clip(hu_image, -1000, 400)
    normalized = (normalized + 1000) / 1400
    input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
    input_resized = F.interpolate(input_tensor, size=(128, 128), mode="bilinear",
                                   align_corners=False)

    with torch.no_grad():
        pred_mask = model(input_resized)

    pred_mask_original_size = F.interpolate(pred_mask, size=original_shape,
                                             mode="bilinear", align_corners=False)
    lung_mask = (pred_mask_original_size.squeeze().numpy() >= 0.5).astype(int)

    lung_pixel_count = lung_mask.sum()
    if lung_pixel_count < 5000:
        from full_pipeline import PipelineError
        raise PipelineError(
            f"Lung mask too small ({lung_pixel_count} pixels) — "
            "no lung tissue detected. Please upload a chest CT scan."
        )

    return lung_mask


def get_resnet_model():
    """
    Load the trained ResNet-18 nodule classifier once and cache it.
    Proven to outperform SimpleCNN on unseen data (87.8% vs 77.7% accuracy,
    see train_resnet.py's docstring for the full comparison).
    """
    if _resnet_model_cache["model"] is None:
        from models.resnet_transfer import build_resnet_model
        model = build_resnet_model()
        checkpoint = torch.load("checkpoints/resnet18_012.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _resnet_model_cache["model"] = model
        print(f"      [resnet] Loaded checkpoint from epoch {checkpoint['epoch']}, "
              f"val_loss {checkpoint['val_loss']:.4f}")
    return _resnet_model_cache["model"]


def find_candidates_with_resnet(hu_image, lung_mask, threshold=0.5):
    """
    Same hole-based candidate detection as find_candidates_with_model(),
    but scored with ResNet-18 instead of SimpleCNN. Same caveat applies:
    only works correctly with classical segment_lungs(), NOT with
    segment_lungs_with_unet()'s cleaner masks.
    """
    model = get_resnet_model()

    filled_lung = ndimage.binary_fill_holes(lung_mask)
    candidate_mask = filled_lung.astype(int) - lung_mask.astype(int)
    labeled = measure.label(candidate_mask)
    regions = measure.regionprops(labeled, intensity_image=hu_image)

    candidates = []
    for region in regions:
        if region.area < 5 or region.area > 1000:
            continue

        y, x = region.centroid
        patch = extract_patch(hu_image, y, x)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        patch_resized = F.interpolate(patch_tensor, size=(224, 224),
                                       mode="bilinear", align_corners=False)

        with torch.no_grad():
            logit = model(patch_resized)
            confidence = torch.sigmoid(logit).item()

        if confidence < threshold:
            continue

        candidates.append({
            'x': x, 'y': y,
            'area': region.area,
            'mean_intensity': region.mean_intensity,
            'confidence': confidence,
        })

    return pd.DataFrame(candidates,
                         columns=['x', 'y', 'area', 'mean_intensity', 'confidence']) \
        if candidates else pd.DataFrame(
            columns=['x', 'y', 'area', 'mean_intensity', 'confidence'])


def find_candidates_density_based(hu_image, lung_mask, threshold=0.5, density_threshold=-400):
    """
    Segmentation-agnostic candidate detection: finds dense regions
    DIRECTLY inside the lung mask, rather than relying on 'holes' created
    by imperfect segmentation. Works correctly regardless of whether
    lung_mask came from the classical threshold method or U-Net's clean
    segmentation — use this when pairing with segment_lungs_with_unet().
    Scores candidates using ResNet-18.
    """
    dense_regions = (hu_image > density_threshold) & (lung_mask > 0)
    labeled = measure.label(dense_regions)
    regions = measure.regionprops(labeled, intensity_image=hu_image)

    model = get_resnet_model()

    candidates = []
    for region in regions:
        if region.area < 5 or region.area > 1000:
            continue

        y, x = region.centroid
        patch = extract_patch(hu_image, y, x)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        patch_resized = F.interpolate(patch_tensor, size=(224, 224),
                                       mode="bilinear", align_corners=False)

        with torch.no_grad():
            confidence = torch.sigmoid(model(patch_resized)).item()

        if confidence < threshold:
            continue

        candidates.append({
            'x': x, 'y': y,
            'area': region.area,
            'mean_intensity': region.mean_intensity,
            'confidence': confidence,
        })

    return pd.DataFrame(candidates,
                         columns=['x', 'y', 'area', 'mean_intensity', 'confidence']) \
        if candidates else pd.DataFrame(
            columns=['x', 'y', 'area', 'mean_intensity', 'confidence'])

def build_tracks(slice_candidates_by_z, track_distance_threshold=20):
    """
    Groups per-slice candidates into tracks based on spatial proximity
    across consecutive slices. See evaluate_multislice_v2.py for the
    original experiment that validated this approach.
    """
    sorted_zs = sorted(slice_candidates_by_z.keys())
    active_tracks = []
    finished_tracks = []

    for z in sorted_zs:
        candidates = slice_candidates_by_z[z]
        matched_track_indices = set()
        for cx, cy, conf in candidates:
            best_track_idx = None
            best_dist = track_distance_threshold
            for idx, track in enumerate(active_tracks):
                if idx in matched_track_indices:
                    continue
                last_z, last_x, last_y, _ = track[-1]
                if z - last_z > 1:
                    continue
                dist = np.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_track_idx = idx
            if best_track_idx is not None:
                active_tracks[best_track_idx].append((z, cx, cy, conf))
                matched_track_indices.add(best_track_idx)
            else:
                active_tracks.append([(z, cx, cy, conf)])

        still_active = []
        for idx, track in enumerate(active_tracks):
            if idx in matched_track_indices or track[-1][0] == z:
                still_active.append(track)
            else:
                finished_tracks.append(track)
        active_tracks = still_active

    finished_tracks.extend(active_tracks)
    return finished_tracks


def analyze_volume_with_multislice_filter(volume, min_consecutive=2, high_conf_bypass=0.9,
                                           confidence_threshold=0.5, density_threshold=-400):
    """
    Full-volume pipeline entry point (NOT for single-slice DICOM uploads).
    Takes a full 3D CT volume (z, y, x), runs U-Net + density-based +
    ResNet-18 detection per slice, then applies the winning multi-slice
    consistency filter validated in evaluate_multislice_v2.py:
        - keep candidates persisting across 2+ consecutive slices
        - OR any single-slice detection with confidence >= 0.9
    This reduced false positives 29% (25.9->18.5 per scan) with ZERO
    sensitivity loss (88.0%) on a 15-scan validation test.

    Returns a DataFrame: x, y, z, confidence, n_slices_tracked
    """
    model = get_resnet_model()
    slice_candidates_by_z = {}

    for z in range(volume.shape[0]):
        hu_image = volume[z].astype(float)
        try:
            lung_mask = segment_lungs_with_unet(hu_image, None)
        except Exception:
            continue

        dense_regions = (hu_image > density_threshold) & (lung_mask > 0)
        labeled = measure.label(dense_regions)
        regions = measure.regionprops(labeled, intensity_image=hu_image)

        candidates = []
        for region in regions:
            if region.area < 5 or region.area > 1000:
                continue
            y, x = region.centroid
            patch = extract_patch(hu_image, y, x)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
            patch_resized = F.interpolate(patch_tensor, size=(224, 224),
                                           mode="bilinear", align_corners=False)
            with torch.no_grad():
                confidence = torch.sigmoid(model(patch_resized)).item()
            if confidence >= confidence_threshold:
                candidates.append((x, y, confidence))
        slice_candidates_by_z[z] = candidates

    tracks = build_tracks(slice_candidates_by_z)
    surviving_tracks = [t for t in tracks if len(t) >= min_consecutive]
    for track in tracks:
        if len(track) < min_consecutive:
            if any(conf >= high_conf_bypass for z, x, y, conf in track):
                surviving_tracks.append(track)

    results = []
    for track in surviving_tracks:
        best = max(track, key=lambda r: r[3])
        z, x, y, conf = best
        results.append({'x': x, 'y': y, 'z': z, 'confidence': conf,
                         'n_slices_tracked': len(track)})

    return pd.DataFrame(results,
                         columns=['x', 'y', 'z', 'confidence', 'n_slices_tracked'])