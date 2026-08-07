# 🫁 AI-Based Lung Nodule Detection

AI-powered lung nodule detection, malignancy estimation, and lung segmentation from chest CT scans. Built as an end-to-end learning project combining classical image processing with trained deep learning models — real data, real training, real evaluation, real bugs found and fixed along the way.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What This Project Actually Does

Given a chest CT scan (DICOM format — single slice or full multi-slice series), this system:

1. **Segments lung tissue** using either a classical HU-threshold method or a trained U-Net (Dice 0.81)
2. **Detects nodule candidates** using density-based analysis inside the lung region
3. **Classifies candidates** as likely nodule vs. not, using a trained CNN — either a compact SimpleCNN or a fine-tuned ResNet-18 (ResNet-18 proven meaningfully better on held-out data)
4. **Estimates malignancy** for detected nodules using a 3D CNN trained on LIDC-IDRI multi-radiologist consensus labels
5. **Maps malignancy scores to illustrative Lung-RADS categories**
6. **Applies multi-slice consistency filtering** for full-volume nodule detection — reduces false positives with no loss in sensitivity, and is available directly in the web app

Every number below is a real, measured result from actual training runs and evaluations — not a placeholder.

---

## 📊 Real, Measured Results

### Nodule Detection (2D classifiers, LUNA16)

Both models trained on subset0+1+2, evaluated on **subset3 — genuinely unseen data**:

| Model | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| SimpleCNN (custom 2-conv-layer CNN) | 77.7% | 71.5% | 81.0% |
| **ResNet-18 (ImageNet transfer learning)** | **87.8%** | **83.5%** | **90.0%** |

SimpleCNN's **FROC score** (LUNA16's standard competition metric — average sensitivity across 7 standard false-positive-rate operating points): **76.6%**

**Finding:** Transfer learning from ImageNet gave a real, meaningful improvement despite the domain gap between natural photos and CT scans.

### Full-Pipeline Detection (U-Net + density-based detection + ResNet-18)

Tested on 15 real LUNA16 scans with 25 documented nodules:

| Confidence threshold | Sensitivity | Avg false positives/scan |
|---|---|---|
| 0.30 | 96.0% | 10.8 |
| 0.50 | 96.0% | 8.3 |
| 0.70 | 88.0% | 6.3 |
| 0.90 | 80.0% | 4.7 |
| 0.99 | 64.0% | 2.1 |

**Multi-slice consistency filtering** (requiring a candidate to appear across 2+ consecutive slices, with a bypass for very high-confidence ≥0.9 single-slice detections) reduced false positives by **29% (25.9→18.5 per scan) with zero sensitivity loss (88.0%)** compared to unfiltered single-slice detection, on a separate 15-scan test.

This filter is implemented as `analyze_volume_with_multislice_filter()` and is **integrated directly into the Streamlit app's "Full Volume Analysis" section** — users can upload an entire DICOM series (all slices of one scan) and get filtered, multi-slice-verified nodule detections, not just single-slice results. Verified end-to-end on a real 40-slice CT volume: correctly built a `(40, 512, 512)` volume, found 27 filtered candidates, several confirmed via genuine multi-slice tracking (`n_slices_tracked` 2–6), with a visually verified detection on real lung anatomy.

### Malignancy Classification (3D CNN, LIDC-IDRI)

Trained on 75 patients (175 nodule annotations, multi-radiologist consensus labels), tested on a held-out 35-sample split:
- **Accuracy: 80.0%**
- Well-calibrated: prediction mean (0.367) closely tracks the true positive rate (0.371)

**Integration status:** Full-volume analysis now uses REAL 3D patches extracted directly from the uploaded volume for malignancy scoring (`estimate_malignancy_3d()`) — verified working on the same 40-slice test scan, producing sensible, non-extreme scores (0.30-0.36 range) across 27 detected candidates. Single-slice mode still uses a 2D-approximated input (a slice stacked into a fake 3D shape), since a single slice has no real depth information to extract a genuine 3D patch from.

**A real bug was found and fixed here**: an initial version had a coordinate-axis ordering error in 3D patch extraction, silently producing empty/padding patches for most nodules. This was caught by noticing a suspicious 100% validation accuracy on only 12 samples — a red flag investigated rather than accepted. After fixing the bug and expanding from 25 to 75 patients, this became a properly diagnosed, trustworthy result.

### Lung Segmentation (U-Net)

- **Dice score: 0.8147**
- **IoU: 0.7858**
- Visually confirmed to produce more anatomically accurate lung boundaries than the classical threshold method on certain scans, where the classical approach over-included non-lung tissue due to its simple intensity threshold.

### Ensemble (3× SimpleCNN, different random seeds)

- 77.0% accuracy vs. 75.1% for the single best model — a modest, real improvement. Model disagreement (uncertainty) was fairly low (std ≈ 0.077), suggesting the three models learned similar patterns rather than being highly diverse.

### Model Export & Optimization

- **PyTorch model**: 804,193 parameters, 3.22 MB
- **ONNX export**: 2.76 KB (successfully converted, opset 17→18)
- **Dynamic INT8 quantization**: did NOT reliably speed up inference for this small model (0.42x–1.06x across three runs — essentially a wash or slightly slower). This is an honest, documented finding: quantization benefits are more pronounced on larger models than the uplift seen here.

---

## 🏗️ System Architecture

```
DICOM File (.dcm) — single slice OR full multi-slice series
        │
        ▼
[1] Load & Preprocess
    → Read DICOM/volume, convert to Hounsfield Units
    → For multi-slice uploads: sort by ImagePositionPatient, stack into 3D volume
    → Validate metadata (RescaleSlope/Intercept), flag unusual slice thickness
        │
        ▼
[2] Lung Segmentation
    → Classical: HU threshold (<-400) + morphological cleanup
    → OR trained U-Net (MiniUNet, Dice 0.81) — recommended, more accurate
        │
        ▼
[3] Candidate Detection
    → Density-based: finds dense tissue regions inside the lung mask
      (segmentation-agnostic — works correctly with either segmentation method)
        │
        ▼
[4] Nodule Classification
    → SimpleCNN or ResNet-18 (recommended — proven better on unseen data)
    → Full-volume uploads: multi-slice consistency filtering applied
      (2+ consecutive slices required, OR confidence ≥0.9 bypass)
      → 29% fewer false positives, zero sensitivity loss
        │
        ▼
[5] Malignancy Estimation
    → 3D CNN + clinical features (diameter, subtlety, texture)
    → Maps to illustrative Lung-RADS category
    → Full-volume mode: REAL 3D patches from the actual volume
    → Single-slice mode: 2D-approximated (fake 3D via slice stacking)
        │
        ▼
┌─────────────────┐    ┌─────────────────┐
│  Streamlit App  │    │   FastAPI REST   │
│  (single-slice  │    │   (Machine API)  │
│  + full-volume) │    │                  │
└─────────────────┘    └─────────────────┘
```

---

## 🛠️ Complete Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10 |
| Deep Learning | PyTorch |
| Medical Imaging | pydicom, SimpleITK |
| Image Processing | scikit-image, SciPy |
| LIDC-IDRI Annotations | pylidc |
| Data Analysis | pandas, NumPy |
| Visualization | Matplotlib |
| Web Interface | Streamlit |
| REST API | FastAPI + Uvicorn |
| Model Export | ONNX |
| GPU Training | Google Colab (T4) |

---

## 🚀 Quick Start

### Prerequisites
- Anaconda or Miniconda
- A chest CT scan in DICOM format (.dcm) — single slice or full series

### Setup
```bash
conda create -n lungcancer python=3.10
conda activate lungcancer
pip install -r requirements.txt
pip install SimpleITK
```

### Run the web app
```bash
streamlit run src/app/app.py
```
- **Single-slice analysis**: upload one `.dcm` file for quick nodule detection + 2D-approximated malignancy estimate
- **Full-volume analysis**: upload all `.dcm` slices of one scan together for multi-slice-filtered nodule detection (takes several minutes on CPU — depends on scan size, verified working on a real 40-slice scan)

### Run the REST API
```bash
python src/app/api.py
# Visit http://localhost:8000/docs for interactive documentation
```

### Run the full pipeline directly
```bash
python full_pipeline.py
```

---

## 📁 Project Structure

```
lung-cancer-detection/
├── data/raw/                        ← Sample DICOM file
├── checkpoints/                     ← Trained model weights
│   ├── best_model.pt                ← SimpleCNN (CPU, subset0+1)
│   ├── best_model_gpu.pt            ← SimpleCNN (GPU, subset0+1+2)
│   ├── resnet18_012.pt              ← ResNet-18 (best nodule classifier)
│   ├── malignancy_model_100patients.pt  ← Malignancy classifier (75 patients)
│   └── unet_model_40ep.pt           ← U-Net segmentation (Dice 0.81)
├── outputs/                         ← Generated visualizations, reports
├── src/
│   ├── preprocessing/
│   │   ├── luna16_dataset.py        ← LUNA16 patch dataset loader
│   │   ├── precompute_luna16_patches.py  ← Efficient 2D patch extraction
│   │   ├── precompute_lung_masks.py ← U-Net training data extraction
│   │   ├── lidc_extraction.py       ← LIDC-IDRI malignancy data extraction
│   │   └── model_integration.py     ← Wires trained models into the pipeline,
│   │                                   including full-volume multi-slice analysis
│   ├── models/
│   │   ├── cnn_2d.py                ← SimpleCNN
│   │   ├── cnn_3d.py                ← 3D CNN (architecture fixed, untrained)
│   │   ├── resnet_transfer.py       ← ResNet-18 transfer learning
│   │   ├── unet.py                  ← MiniUNet segmentation
│   │   ├── malignancy_classifier.py ← 3D CNN + clinical features
│   │   └── ensemble.py              ← Multi-model averaging
│   ├── training/
│   │   ├── train_nodule_gpu.py      ← SimpleCNN training
│   │   ├── train_resnet.py          ← ResNet-18 training
│   │   ├── train_unet.py            ← U-Net training (Dice/IoU tracking)
│   │   └── train_malignancy.py      ← Malignancy classifier training
│   ├── evaluation/
│   │   ├── clinical_metrics.py      ← Sensitivity, specificity, PPV, NPV
│   │   ├── froc_real.py             ← Real FROC evaluation
│   │   ├── grad_cam.py              ← Grad-CAM on real trained checkpoints
│   │   └── lung_rads.py             ← Malignancy → Lung-RADS mapping
│   ├── optimization/
│   │   └── model_optimizer.py       ← ONNX export, quantization benchmarking
│   └── app/
│       ├── app.py                   ← Streamlit interface (single-slice + full-volume)
│       └── api.py                   ← FastAPI REST endpoint
├── full_pipeline.py                 ← End-to-end integrated pipeline
├── requirements.txt
└── README.md
```

---

## 🔬 What Was Actually Learned (Debugging Log)

This project involved real, non-trivial debugging — worth documenting honestly:

- **Coordinate axis-order bug** in 3D patch extraction for the malignancy classifier — caused by mismatching `pylidc`'s centroid convention, silently producing empty patches. Caught by noticing a suspiciously perfect validation score, not by the code failing outright.
- **Memory leak** during LUNA16 patch precomputation — caused by SimpleITK volume objects not being released properly when cached across many candidates. Fixed by explicit per-scan loading with forced garbage collection.
- **Segmentation/candidate-detection coupling bug** — the classical candidate-finding method secretly relied on "holes" created by the classical segmentation's imperfections. Switching to U-Net's cleaner masks broke this silently (found 0 candidates on a scan with a confirmed nodule) until candidate detection was redesigned to work directly on tissue density rather than mask holes.
- **Quantization does not always help** — dynamic INT8 quantization was tested honestly and found not to improve inference speed for this project's small model size, rather than force-reporting a positive result.
- **Multi-slice filter tuning** — an initial 3-consecutive-slice requirement cut false positives sharply but cost too much sensitivity (96%→64%); a high-confidence bypass rule (keep single-slice detections ≥0.9 confidence even without multi-slice consistency) recovered full sensitivity while still cutting false positives 29%.

---

## ⚠️ Limitations & Honest Scope

### What this project does NOT do
- **Does not determine cancer stage.** Staging requires lymph node imaging (often PET), whole-body metastasis screening, and frequently tissue biopsy — none of which a chest CT nodule detector can provide. This is a fundamental data-availability limitation, not something more training data or bigger models would fix.
- **Malignancy scoring is 2D-approximated in single-slice mode only.** Full-volume mode now uses real 3D patches extracted from the actual uploaded volume (verified working). Single-slice mode still approximates by stacking one 2D slice into a fake 3D shape, since a single slice has no genuine depth to extract a real patch from — clearly caveated in the UI. Even with real 3D patches, malignancy scores use default (neutral 0.5) subtlety/texture clinical features, since these come from radiologist ratings not available outside LIDC-IDRI-style annotations — so full-volume malignancy scores are more genuine than single-slice, but still not clinically validated.
- **Full-volume nodule detection is slow on CPU** — several minutes per scan depending on slice count, since every slice runs full segmentation + detection + classification. Genuinely usable, just not fast; a GPU (e.g. via Colab) would make this dramatically faster, following the same speedup seen during training (roughly 1000x for the 2D classifier).

### Known dataset/scale constraints
- Malignancy classifier trained on 75 patients — small by research standards; more patients would likely improve reliability further.
- Nodule classifiers trained on 3 of LUNA16's 10 subsets — full-dataset training would likely improve results further, following the same "more data helps" pattern observed throughout this project.
- U-Net is a small architecture (a few conv layers) — a deeper network with more training data would likely exceed Dice 0.81.

### Regulatory Status
This project is **NOT** FDA approved or CE marked and has not undergone clinical validation trials of any kind.

### Clinical Disclaimer
> ⚠️ **For research and educational purposes only.**
> This tool is NOT approved for clinical diagnosis.
> All predictions must be reviewed by a qualified radiologist.
> Do not use for actual patient care decisions.

---

## 📚 References & Datasets

| Resource | Link |
|----------|------|
| LUNA16 Dataset | https://luna16.grand-challenge.org/ |
| LIDC-IDRI Dataset | https://www.cancerimagingarchive.net/collection/lidc-idri/ |
| pylidc | https://pylidc.github.io/ |
| pydicom | https://pydicom.github.io/ |
| Lung-RADS Guidelines | https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-RADS |

---

## 📄 License

This project is licensed under the MIT License.