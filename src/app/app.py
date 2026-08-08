import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pydicom
import pandas as pd
import tempfile
import os
import sys
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from full_pipeline import run_full_pipeline
import torch
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'preprocessing'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation'))

from models.malignancy_classifier import MalignancyClassifier
from evaluation.lung_rads import malignancy_to_lungrads

@st.cache_resource
def load_malignancy_model():
    model = MalignancyClassifier()
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'checkpoints', 'malignancy_model_100patients.pt'
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def estimate_malignancy(hu_image, candidate_x, candidate_y, area):
    """
    Rough malignancy estimate using a 2D candidate location.
    NOTE: MalignancyClassifier expects a real 3D patch + 3 clinical
    features (diameter, subtlety, texture) from LIDC-IDRI-style data.
    Since full_pipeline.py only gives us a single 2D slice, we build an
    approximate 3D patch by stacking the same 2D slice, and use only
    diameter as a real feature (subtlety/texture default to a neutral 0.5
    since we don't have radiologist-style ratings in this 2D pipeline).
    This is a simplification — a true 3D scan would give a more accurate
    input to this model.
    """
    model = load_malignancy_model()

    half = 16
    y, x = int(candidate_y), int(candidate_x)
    y_min, y_max = max(0, y - half), min(hu_image.shape[0], y + half)
    x_min, x_max = max(0, x - half), min(hu_image.shape[1], x + half)

    patch_2d = hu_image[y_min:y_max, x_min:x_max]
    patch_2d = np.clip(patch_2d, -1000, 400)
    patch_2d = (patch_2d + 1000) / 1400

    if patch_2d.shape != (32, 32):
        pad_y = 32 - patch_2d.shape[0]
        pad_x = 32 - patch_2d.shape[1]
        patch_2d = np.pad(patch_2d, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

    patch_3d = np.stack([patch_2d] * 32, axis=0).astype(np.float32)
    patch_tensor = torch.from_numpy(patch_3d).unsqueeze(0).unsqueeze(0)

    diameter_estimate = np.sqrt(area / np.pi) * 2
    clinical_features = torch.tensor([[diameter_estimate, 0.5, 0.5]], dtype=torch.float32)

    with torch.no_grad():
        malignancy_score = model(patch_tensor, clinical_features).item()

    return malignancy_score


def build_volume_from_dicom_series(dicom_files):
    """
    Takes a list of uploaded DICOM slice files, sorts them into correct
    order, and builds a 3D HU volume + returns per-slice metadata.
    """
    slices = []
    for f in dicom_files:
        ds = pydicom.dcmread(f, force=True)
        slices.append(ds)

    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda s: int(s.InstanceNumber))

    volume = np.stack([s.pixel_array for s in slices])
    slope = float(slices[0].RescaleSlope)
    intercept = float(slices[0].RescaleIntercept)
    hu_volume = volume * slope + intercept

    return hu_volume, slices


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Nodule Detection AI",
    page_icon="🫁",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────
st.title("🫁 AI-based Lung Nodule Detection")
st.markdown("*AI-powered nodule candidate detection from CT scans*")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.3,
        value=0.1,
        step=0.01,
        help="Higher threshold = fewer but more confident detections"
    )
    st.divider()
    st.markdown("**About this tool:**")
    st.markdown("This AI tool assists radiologists in detecting suspicious nodule candidates in chest CT scans.")
    st.warning("⚠️ For research use only. Not for clinical diagnosis.")

# ── File Upload ──────────────────────────────────────────────
st.header("📁 Upload CT Scan")
uploaded_file = st.file_uploader(
    "Upload a DICOM file (.dcm)",
    type=['dcm'],
    help="Upload a chest CT scan in DICOM format"
)
@st.cache_data
def run_catched_pipeline(file_bytes,threshold):
    """Cache pipeline results so same file+threshold doesn't reprocess"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path=tmp_file.name
    try:
        candidates_df,warnings=run_full_pipeline(tmp_path,threshold)
        return candidates_df,warnings,tmp_path
    except Exception as e:
        raise e

if uploaded_file is not None:
    
    
    
    # Run pipeline with caching
    with st.spinner("🔄 Processing CT scan... please wait"):
        try:
            candidates_df, warnings, tmp_path = run_catched_pipeline(uploaded_file.getvalue(),threshold)
            st.success("✅ Pipeline complete!")
        except Exception as e:
            st.error(f"❌ Pipeline failed: {str(e)}")
            st.stop()
    
    # Show warnings if any
    if warnings.has_warnings():
        for w in warnings.warnings:
            st.warning(f"⚠️ {w}")
    
    # Load DICOM for display
    ds = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()), force=True)
    pixel_array = ds.pixel_array
    slope = float(ds.RescaleSlope)
    intercept = float(ds.RescaleIntercept)
    hu_image = np.clip(pixel_array * slope + intercept, -1000, 400)
    
    st.divider()
    # ── Malignancy Estimation ──────────────────────────────
    if not candidates_df.empty:
        malignancy_scores = []
        lungrads_categories = []
        for _, row in candidates_df.iterrows():
            score = estimate_malignancy(hu_image, row['x'], row['y'], row['area'])
            malignancy_scores.append(score)
            lungrads_categories.append(malignancy_to_lungrads(score))
        candidates_df = candidates_df.copy()
        candidates_df['malignancy_score'] = malignancy_scores
        candidates_df['lung_rads'] = lungrads_categories
    
    # ── Results Layout ────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Candidates Found", len(candidates_df))
    with col2:
        st.metric("Warnings", len(warnings.warnings))
    with col3:
        st.metric("Scan Shape", f"{pixel_array.shape[0]}×{pixel_array.shape[1]}")
    
    st.divider()
    
    # ── CT Scan Display ───────────────────────────────────────
    st.header("🔬 CT Scan Analysis")
    
    img_col, report_col = st.columns([2, 1])
    
    with img_col:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original CT
        axes[0].imshow(hu_image, cmap='gray')
        axes[0].set_title('CT Scan (HU)')
        axes[0].axis('off')
        
        # CT with candidates marked
        axes[1].imshow(hu_image, cmap='gray')
        if not candidates_df.empty:
            axes[1].scatter(
                candidates_df['x'],
                candidates_df['y'],
                c='red', s=100, marker='x',
                linewidths=2, label='Candidates'
            )
            # Draw circles around candidates
            for _, row in candidates_df.iterrows():
                circle = plt.Circle(
                    (row['x'], row['y']),
                    radius=15,
                    color='red',
                    fill=False,
                    linewidth=2
                )
                axes[1].add_patch(circle)
        
        axes[1].set_title(f'Detected Candidates ({len(candidates_df)})')
        axes[1].axis('off')
        if not candidates_df.empty:
            axes[1].legend(loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with report_col:
        st.subheader("📋 Candidates Report")
        
        if candidates_df.empty:
            st.info("No candidates found above threshold.")
        else:
            # Display candidates table
            display_df = candidates_df.copy()
            display_df['x'] = display_df['x'].round(1)
            display_df['y'] = display_df['y'].round(1)
            display_df['area'] = display_df['area'].astype(int)
            display_df['mean_intensity'] = display_df['mean_intensity'].round(1)
            if 'malignancy_score' in display_df.columns:
                display_df['malignancy_score'] = display_df['malignancy_score'].round(3)
            st.dataframe(display_df, use_container_width=True)

            if 'malignancy_score' in candidates_df.columns:
                st.caption(
                    "⚠️ Malignancy scores are approximated from a single 2D slice — "
                    "the underlying model was trained on full 3D CT volumes. "
                    "Treat these as illustrative, not clinically accurate."
                )
            # Download button
            csv = candidates_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name="nodule_candidates.csv",
                mime="text/csv"
            )
    
    st.divider()
    
    # ── Patient Info ──────────────────────────────────────────
    st.header("👤 Patient Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Patient ID",
                 str(ds.get('PatientID', 'N/A')))
    with info_col2:
        st.metric("Patient Age",
                 str(ds.get('PatientAge', 'N/A')))
    with info_col3:
        st.metric("Scanner",
                 str(ds.get('Manufacturer', 'N/A')))
    
    # Cleanup temp file
    try:
      os.unlink(tmp_path)

    except(FileNotFoundError, TypeError):
        pass

else:
    # Show instructions when no file uploaded
    st.info("👆 Upload a DICOM (.dcm) file to begin analysis")
    
    st.markdown("""
    ### How to use this tool:
    1. **Upload** a chest CT scan in DICOM format
    2. **Wait** for the AI pipeline to process the scan
    3. **Review** the detected nodule candidates
    4. **Download** the CSV report for your records
    
    ### What this tool detects:
    - Suspicious nodule candidates inside lung tissue
    - Candidates are marked with red circles on the CT image
    - Each candidate includes position, size, and density information
    """)

# ── Full Volume Analysis (Multi-Slice) ──────────────────────
st.divider()
st.header("🧊 Full Volume Analysis (Multi-Slice)")
st.markdown(
    "Upload **all DICOM slices** from one scan for more accurate detection "
    "using multi-slice consistency filtering — reduces false positives by "
    "~29% compared to single-slice analysis, with no loss in sensitivity "
    "(validated on 15 LUNA16 scans)."
)
st.warning(
    "⏱️ This processes every slice individually and can take **several minutes** "
    "on CPU, depending on scan size (typically 100-300 slices)."
)

volume_files = st.file_uploader(
    "Upload all DICOM slices for one scan",
    type=['dcm'],
    accept_multiple_files=True,
    key="volume_uploader"
)

if volume_files and len(volume_files) > 1:
    if st.button("🚀 Run Full Volume Analysis"):
        from preprocessing.model_integration import analyze_volume_with_multislice_filter

        with st.spinner(f"Building volume from {len(volume_files)} slices..."):
            hu_volume, slices_meta = build_volume_from_dicom_series(volume_files)
            st.info(f"Volume shape: {hu_volume.shape}")

        with st.spinner("Running slice-by-slice analysis (this takes several minutes)..."):
            results = analyze_volume_with_multislice_filter(hu_volume)

        if not results.empty:
            from preprocessing.model_integration import estimate_malignancy_3d, get_malignancy_model

            with st.spinner("Estimating malignancy from real 3D patches..."):
                malignancy_scores = []
                lungrads_categories = []
                for _, row in results.iterrows():
                    score = estimate_malignancy_3d(hu_volume, row['x'], row['y'], row['z'])
                    if score is None:
                        score = 0.0
                    malignancy_scores.append(score)
                    lungrads_categories.append(malignancy_to_lungrads(score))

                results = results.copy()
                results['malignancy_score'] = malignancy_scores
                results['lung_rads'] = lungrads_categories

        st.success(f"✅ Found {len(results)} candidates with multi-slice filtering")
        if not results.empty:
            st.caption(
                "✅ Malignancy scores here use REAL 3D patches from the uploaded "
                "volume (not the 2D approximation used in single-slice mode)."
            )
        if not results.empty:
            display_results = results.copy()
            display_results['x'] = display_results['x'].round(1)
            display_results['y'] = display_results['y'].round(1)
            display_results['confidence'] = display_results['confidence'].round(3)
            if 'malignancy_score' in display_results.columns:
                display_results['malignancy_score'] = display_results['malignancy_score'].round(3)
            st.dataframe(display_results, use_container_width=True)
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Full-Volume Results CSV",
                data=csv,
                file_name="volume_candidates.csv",
                mime="text/csv"
            )

            top_candidate = results.loc[results['confidence'].idxmax()]
            top_z = int(top_candidate['z'])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(hu_volume[top_z], cmap='gray')
            ax.scatter([top_candidate['x']], [top_candidate['y']], c='red', s=100, marker='x')
            ax.set_title(f"Top candidate (slice {top_z}, confidence {top_candidate['confidence']:.3f})")
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No candidates found after multi-slice filtering.")