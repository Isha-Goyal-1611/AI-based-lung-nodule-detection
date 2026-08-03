import numpy as np
import torch
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from preprocessing.model_integration import find_candidates_density_based, segment_lungs_with_unet
# ── Error Classes ────────────────────────────────────────────
class PipelineError(Exception):
    """Raised when pipeline cannot continue"""
    pass

class PipelineWarning:
    """Non-fatal issues that should be flagged to user"""
    def __init__(self):
        self.warnings = []
    
    def add(self, message):
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
    
    def has_warnings(self):
        return len(self.warnings) > 0

# ── Step 1: Load DICOM ───────────────────────────────────────
def load_dicom(dicom_path, warnings):
    """Load DICOM with proper error handling"""
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
    except Exception as e:
        raise PipelineError(f"Cannot read DICOM file: {e}")
    
    # Check required metadata
    if not hasattr(ds, 'RescaleSlope') or not hasattr(ds, 'RescaleIntercept'):
        raise PipelineError(
            "Required metadata missing: RescaleSlope/RescaleIntercept not found. "
            "Cannot convert to Hounsfield Units."
        )
    
    # Check slice thickness
    slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
    if slice_thickness > 3.0:
        warnings.add(
            f"Slice thickness = {slice_thickness}mm "
            f"(model trained on ≤3mm scans — predictions may be less accurate)"
        )
    
    pixel_array = ds.pixel_array
    slope = float(ds.RescaleSlope)
    intercept = float(ds.RescaleIntercept)
    
    return pixel_array, slope, intercept, ds

# ── Step 2: Preprocess ───────────────────────────────────────
def preprocess(pixel_array, slope, intercept):
    """Convert to HU, clip, normalize"""
    hu_image = pixel_array * slope + intercept
    hu_clipped = np.clip(hu_image, -1000, 400)
    normalized = (hu_clipped + 1000) / 1400
    return hu_image, normalized

# ── Step 3: Segment Lungs (CLASSICAL — kept for reference/comparison) ──
def segment_lungs(hu_image, warnings):
    """Extract lung mask with validation (classical threshold method)"""
    binary_mask = hu_image < -400
    cleaned = morphology.binary_erosion(binary_mask)
    cleaned = morphology.binary_dilation(cleaned)
    labeled = measure.label(cleaned)
    regions = measure.regionprops(labeled)
    
    if len(regions) == 0:
        raise PipelineError(
            "No regions found after segmentation — "
            "this may not be a chest CT scan."
        )
    
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
    top_2 = sorted_regions[:2]
    
    lung_mask = np.zeros_like(labeled)
    for region in top_2:
        lung_mask[labeled == region.label] = 1
    
    # Validate lung mask size
    lung_pixel_count = lung_mask.sum()
    if lung_pixel_count < 5000:
        raise PipelineError(
            f"Lung mask too small ({lung_pixel_count} pixels) — "
            "no lung tissue detected. Please upload a chest CT scan."
        )
    
    return lung_mask

# ── Step 4: Find Candidates ──────────────────────────────────
def find_candidates(hu_image, lung_mask, threshold=0.5):
    """Find nodule candidates using fill-holes method (classical, kept for reference)"""
    filled_lung = ndimage.binary_fill_holes(lung_mask)
    candidate_mask = filled_lung.astype(int) - lung_mask.astype(int)
    labeled = measure.label(candidate_mask)
    regions = measure.regionprops(labeled, intensity_image=hu_image)
    
    candidates = []
    for region in regions:
        if region.area < 5 or region.area > 1000:
            continue
        confidence = (region.mean_intensity + 400)/800
        confidence = max(0.0, min(1.0,confidence))
        
        if confidence<threshold:
            continue
        y, x = region.centroid
        candidates.append({
            'x': x, 'y': y,
            'area': region.area,
            'mean_intensity': region.mean_intensity,
            'confidence':confidence
        })
    
    return pd.DataFrame(candidates,
                       columns=['x','y','area','mean_intensity','confidence']) \
           if candidates else pd.DataFrame(
               columns=['x','y','area','mean_intensity','confidence'])

# ── Step 5: Generate Report ──────────────────────────────────
def generate_report(dicom_path, hu_image, lung_mask,
                   candidates_df, warnings, output_dir='outputs'):
    """Save results and generate visual report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save candidates CSV
    csv_path = os.path.join(output_dir, 'final_candidates.csv')
    candidates_df.to_csv(csv_path, index=False)
    
    # Save visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(hu_image, cmap='gray')
    axes[0].set_title('HU Image')
    
    axes[1].imshow(lung_mask, cmap='gray')
    axes[1].set_title('Lung Mask (U-Net)')
    
    axes[2].imshow(hu_image, cmap='gray')
    if not candidates_df.empty:
        axes[2].scatter(candidates_df['x'], candidates_df['y'],
                       c='red', s=50, marker='x')
    axes[2].set_title(f'{len(candidates_df)} Candidates Found')
    
    img_path = os.path.join(output_dir, 'day22_full_pipeline.png')
    plt.savefig(img_path)
    plt.show()
    
    # Print report
    print("\n" + "="*50)
    print("LUNG CANCER DETECTION PIPELINE REPORT")
    print("="*50)
    print(f"Input file:        {dicom_path}")
    print(f"Candidates found:  {len(candidates_df)}")
    print(f"Warnings:          {len(warnings.warnings)}")
    if warnings.has_warnings():
        for w in warnings.warnings:
            print(f"  ⚠️  {w}")
    print(f"Results saved to:  {output_dir}/")
    print("="*50)
    
    return csv_path, img_path

# ── Main Pipeline ────────────────────────────────────────────
def run_full_pipeline(dicom_path, threshold=0.5):
    """
    Complete end-to-end pipeline
    Returns: (candidates_df, report_paths, warnings)
    """
    warnings = PipelineWarning()
    
    print("🫁 Starting Lung Cancer Detection Pipeline...")
    print(f"   Processing: {dicom_path}")
    
    # Step 1
    print("\n[1/5] Loading DICOM...")
    pixel_array, slope, intercept, ds = load_dicom(dicom_path, warnings)
    print(f"      Shape: {pixel_array.shape}, dtype: {pixel_array.dtype}")
    
    # Step 2
    print("\n[2/5] Preprocessing (HU conversion + normalization)...")
    hu_image, normalized = preprocess(pixel_array, slope, intercept)
    print(f"      HU range: [{hu_image.min():.0f}, {hu_image.max():.0f}]")
    
    # Step 3 — NOW USES TRAINED U-NET instead of classical threshold
    print("\n[3/5] Segmenting lungs (U-Net)...")
    lung_mask = segment_lungs_with_unet(hu_image, warnings)
    print(f"      Lung pixels: {lung_mask.sum():,}")
    
    # Step 4 — uses trained SimpleCNN/ResNet nodule classifier
    print("\n[4/5] Finding nodule candidates...")
    candidates_df = find_candidates_density_based(hu_image, lung_mask, threshold)
    print(f"      Candidates: {len(candidates_df)}")
    
    # Step 5
    print("\n[5/5] Generating report...")
    csv_path, img_path = generate_report(
        dicom_path, hu_image, lung_mask, candidates_df, warnings
    )
    
    print("\n✅ Pipeline complete!")
    return candidates_df, warnings

# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    candidates, warnings = run_full_pipeline('data/raw/sample.dcm', threshold=0.5)
    print(candidates[['x', 'y', 'confidence']])