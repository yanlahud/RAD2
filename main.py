from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from typing import List
import os
import uuid
import numpy as np
import SimpleITK as sitk
from fpdf import FPDF
from ia_infer import run_inference
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

BASE_DIR = "cbct_uploads"
RESULTS_DIR = "cbct_results"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Permitir CORS para o frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-cbct/")
async def upload_cbct(files: List[UploadFile] = File(...)):
    exam_id = str(uuid.uuid4())
    exam_path = os.path.join(BASE_DIR, exam_id)
    os.makedirs(exam_path, exist_ok=True)

    # Salva os arquivos DICOM
    for file in files:
        file_path = os.path.join(exam_path, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

    # Etapa 1 - Conversão em volume
    volume_path = os.path.join(RESULTS_DIR, f"{exam_id}_volume.nii.gz")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(exam_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, volume_path)

    # Etapa 2 - Inferência com MONAI
    output_path = os.path.join(RESULTS_DIR, f"{exam_id}_ia_mask.nii.gz")
    run_inference(volume_path, output_path)

    # Etapa 3 - Gerar Laudo
    laudo_path = os.path.join(RESULTS_DIR, f"{exam_id}_laudo.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"""LAUDO TOMOGRÁFICO ODONTOLÓGICO
ID do Exame: {exam_id}

ACHADOS:
- Segmentação automática com MONAI.
- Detecção volumétrica aplicada com base na máscara.
- (Em breve: interpretação anatômica automatizada).

IMPRESSÃO DIAGNÓSTICA:
Laudo gerado automaticamente por IA.
""")
    pdf.output(laudo_path)

    return {
        "exam_id": exam_id,
        "status": "completo",
        "laudo_url": f"https://rad2.onrender.com/gerar-laudo/{exam_id}"
    }

@app.get("/gerar-laudo/{exam_id}")
def gerar_laudo(exam_id: str):
    laudo_path = os.path.join(RESULTS_DIR, f"{exam_id}_laudo.pdf")
    return FileResponse(laudo_path, filename=f"{exam_id}_laudo.pdf")
