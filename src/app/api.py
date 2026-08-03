import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import uuid
import json
from datetime import datetime
from full_pipeline import run_full_pipeline
import torch
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evaluation'))

from models.malignancy_classifier import MalignancyClassifier
from evaluation.lung_rads import malignancy_to_lungrads

_malignancy_model = None

def get_malignancy_model():
    global _malignancy_model
    if _malignancy_model is None:
        _malignancy_model = MalignancyClassifier()
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'checkpoints', 'malignancy_model_100patients.pt'
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        _malignancy_model.load_state_dict(checkpoint['model_state_dict'])
        _malignancy_model.eval()
    return _malignancy_model


def estimate_malignancy(hu_image, candidate_x, candidate_y, area):
    """
    Approximates malignancy from a 2D slice by stacking it into a fake 3D
    volume. See app.py's estimate_malignancy() for the full explanation
    of this simplification.
    """
    model = get_malignancy_model()

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
        return model(patch_tensor, clinical_features).item()

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Lung Cancer Detection API",
    description="AI-powered lung nodule candidate detection from CT scans",
    version="1.0.0"
)

# In-memory storage for results (in production: use a real database)
results_store = {}

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if API is running"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "message": "Lung Cancer Detection API is running"
    }

@app.post("/analyze")
async def analyze_ct_scan(file: UploadFile = File(...)):
    """
    Upload a DICOM file and get nodule candidates back, including
    malignancy scores and Lung-RADS categories per candidate.
    Returns: scan_id, candidates list, warnings
    """
    import pydicom

    # Step 1: Validate file type
    if not file.filename.endswith('.dcm'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a DICOM (.dcm) file."
        )

    # Step 2: Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    # Step 3: Run pipeline, and separately compute hu_image for malignancy
    # scoring BEFORE the temp file gets deleted.
    try:
        candidates_df, warnings = run_full_pipeline(tmp_path, threshold=0.1)

        ds = pydicom.dcmread(tmp_path, force=True)
        pixel_array = ds.pixel_array
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        hu_image = np.clip(pixel_array * slope + intercept, -1000, 400)

        if not candidates_df.empty:
            candidates_df = candidates_df.copy()
            candidates_df['malignancy_score'] = candidates_df.apply(
                lambda row: estimate_malignancy(hu_image, row['x'], row['y'], row['area']),
                axis=1
            )
            candidates_df['lung_rads'] = candidates_df['malignancy_score'].apply(
                malignancy_to_lungrads
            )
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(
            status_code=422,
            detail=f"Pipeline failed: {str(e)}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Step 4: Generate unique scan ID
    scan_id = str(uuid.uuid4())[:8]

    # Step 5: Prepare results
    results = {
        "scan_id": scan_id,
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "candidates_found": len(candidates_df),
        "warnings": warnings.warnings,
        "malignancy_disclaimer": (
            "Malignancy scores are approximated from a single 2D slice; the "
            "underlying model was trained on full 3D CT volumes. Treat as "
            "illustrative, not clinically accurate."
        ),
        "candidates": candidates_df.to_dict(orient='records')
            if not candidates_df.empty else []
    }

    # Step 6: Store results for later retrieval
    results_store[scan_id] = results

    return JSONResponse(content=results)

@app.get("/results/{scan_id}")
def get_results(scan_id: str):
    """
    Retrieve results of a previously analyzed scan
    """
    if scan_id not in results_store:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for scan_id: {scan_id}"
        )
    return JSONResponse(content=results_store[scan_id])

@app.get("/results")
def list_all_results():
    """
    List all analyzed scans
    """
    return {
        "total_scans": len(results_store),
        "scan_ids": list(results_store.keys()),
        "scans": [
            {
                "scan_id": k,
                "timestamp": v["timestamp"],
                "filename": v["filename"],
                "candidates_found": v["candidates_found"]
            }
            for k, v in results_store.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)