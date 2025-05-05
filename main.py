from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import os
import uuid
import numpy as np
import SimpleITK as sitk

app = FastAPI()

BASE_DIR = "cbct_uploads"
RESULTS_DIR = "cbct_results"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/upload-cbct/")
async def upload_cbct(files: List[UploadFile] = File(...)):
    exam_id = str(uuid.uuid4())
    exam_path = os.path.join(BASE_DIR, exam_id)
    os.makedirs(exam_path, exist_ok=True)

    for file in files:
        file_path = os.path.join(exam_path, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

    return {"id": exam_id, "status": "uploaded", "path": exam_path}

@app.get("/analisar/{exam_id}")
def analisar_cbct(exam_id: str):
    exam_path = os.path.join(BASE_DIR, exam_id)
    if not os.path.exists(exam_path):
        return JSONResponse(status_code=404, content={"error": "Exame não encontrado"})

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(exam_path)
    if not series_ids:
        return JSONResponse(status_code=400, content={"error": "Nenhuma série DICOM encontrada"})

    file_names = reader.GetGDCMSeriesFileNames(exam_path, series_ids[0])
    reader.SetFileNames(file_names)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image)

    mask = (volume > np.percentile(volume, 99)).astype(np.uint8)
    result_path = os.path.join(RESULTS_DIR, f"{exam_id}_mask.npy")
    np.save(result_path, mask)

    return {
        "exam_id": exam_id,
        "volume_shape": list(volume.shape),
        "result_file": result_path
    }

@app.get("/resultado/{exam_id}")
def resultado_cbct(exam_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{exam_id}_mask.npy")
    if not os.path.exists(result_path):
        return JSONResponse(status_code=404, content={"error": "Resultado não encontrado"})

    return {
        "exam_id": exam_id,
        "mask_preview": "Simulado (real volume)",
        "shape": list(np.load(result_path).shape)
    }
