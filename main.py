from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import uuid
import numpy as np
import SimpleITK as sitk
from fpdf import FPDF
from ia_infer import run_inference

app = FastAPI()

# ✅ Middleware de CORS habilitado para qualquer origem (ajuste isso em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção troque por ['https://seusite.vercel.app']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/ia-analisar/{exam_id}")
def ia_analisar_cbct(exam_id: str):
    volume_path = os.path.join(RESULTS_DIR, f"{exam_id}_volume.nii.gz")
    output_path = os.path.join(RESULTS_DIR, f"{exam_id}_ia_mask.nii.gz")

    if not os.path.exists(volume_path):
        return JSONResponse(status_code=404, content={"error": "Volume não encontrado. Execute /analisar primeiro."})

    run_inference(volume_path, output_path)

    return {
        "exam_id": exam_id,
        "ia_mask_file": output_path,
        "message": "Inferência MONAI concluída"
    }

@app.get("/gerar-laudo/{exam_id}")
def gerar_laudo(exam_id: str):
    laudo_path = os.path.join(RESULTS_DIR, f"{exam_id}_laudo.pdf")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"""
LAUDO TOMOGRÁFICO ODONTOLÓGICO
ID do Exame: {exam_id}

ACHADOS:
- Segmentação automatizada realizada com MONAI (modelo cabeça-pescoço).
- Achados estruturais detectados com base em máscara segmentada.
- (Futuro: interpretar máscara para gerar descrição anatômica automática).

IMPRESSÃO DIAGNÓSTICA:
Modelo executado com sucesso. Integração de IA concluída para validação clínica.

--- Laudo gerado automaticamente por IA ---
""")
    pdf.output(laudo_path)

    return FileResponse(laudo_path, filename=f"{exam_id}_laudo.pdf")
