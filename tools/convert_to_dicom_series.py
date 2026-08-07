import SimpleITK as sitk
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import os
import datetime

# Pick a real LUNA16 scan you already have locally
MHD_PATH = r"C:\lung_data\luna16\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"
OUTPUT_DIR = "test_dicom_series"
MAX_SLICES = 40  # limit slices so upload/processing stays manageable

os.makedirs(OUTPUT_DIR, exist_ok=True)

itk_img = sitk.ReadImage(MHD_PATH)
volume = sitk.GetArrayFromImage(itk_img)  # (z, y, x), already in HU for LUNA16
spacing = itk_img.GetSpacing()

print(f"Volume shape: {volume.shape}")

# Pick a window of consecutive slices around the middle (likely to contain
# actual lung anatomy, not just empty space at the very top/bottom)
mid = volume.shape[0] // 2
start = max(0, mid - MAX_SLICES // 2)
end = min(volume.shape[0], start + MAX_SLICES)
print(f"Exporting slices {start} to {end}")

series_uid = generate_uid()
study_uid = generate_uid()

for i, z in enumerate(range(start, end)):
    slice_data = volume[z].astype(np.int16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(f"slice_{i:03d}.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = "Test^Volume"
    ds.PatientID = "TESTVOL001"
    ds.PatientAge = "060Y"
    ds.Modality = "CT"
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.InstanceNumber = i + 1
    ds.ImagePositionPatient = [0, 0, float(z * spacing[2])]
    ds.SliceThickness = str(spacing[2])
    ds.PixelSpacing = [str(spacing[0]), str(spacing[1])]

    ds.RescaleSlope = "1"
    ds.RescaleIntercept = "0"  # LUNA16 volumes are already in HU

    ds.Rows, ds.Columns = slice_data.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed, since HU includes negative values

    ds.PixelData = slice_data.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    out_path = os.path.join(OUTPUT_DIR, f"slice_{i:03d}.dcm")
    ds.save_as(out_path, write_like_original=False)

print(f"\nSaved {end - start} real DICOM slices to {OUTPUT_DIR}/")
print("Upload all files from this folder together in the Full Volume Analysis section.")