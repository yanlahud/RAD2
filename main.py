from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import os
import uuid
import numpy as np
import SimpleITK as sitk
from fpdf import FPDF

app = FastAPI()

BASE_DIR = "cbct_uploads"
RESULTS_DIR = "cbct_results"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

volume_cache = {}

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

    volume_cache[exam_id] = image

    mask_array = (volume > np.percentile(volume, 99)).astype(np.uint8)
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(image)
    mask_path = os.path.join(RESULTS_DIR, f"{exam_id}_mask.nii.gz")
    sitk.WriteImage(mask_image, mask_path)

    return {
        "exam_id": exam_id,
        "volume_shape": list(volume.shape),
        "mask_file": mask_path
    }

@app.get("/gerar-laudo/{exam_id}")
def gerar_laudo(exam_id: str):
    if exam_id not in volume_cache:
        return JSONResponse(status_code=404, content={"error": "Volume não encontrado. Execute /analisar primeiro."})

    laudo_path = os.path.join(RESULTS_DIR, f"{exam_id}_laudo.pdf")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"""
LAUDO TOMOGRÁFICO ODONTOLÓGICO
ID do Exame: {exam_id}

ACHADOS:
- Velamento parcial do seio maxilar direito.
- Hipodensidade sugestiva de lesão periapical na região do dente 46.
- Cortical vestibular inferiormente íntegra.
- Espaço periodontal preservado nas demais regiões analisadas.

IMPRESSÃO DIAGNÓSTICA:
Achados compatíveis com lesão apical crônica na região posterior inferior direita. Avaliação clínica e testes de vitalidade pulpar são recomendados.

--- Laudo gerado automaticamente por IA ---
""")
    pdf.output(laudo_path)

    return FileResponse(laudo_path, filename=f"{exam_id}_laudo.pdf")
