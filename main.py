from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import os
import json
from datetime import datetime
from ia_infer import inferir_cbct
from fpdf import FPDF

app = FastAPI()

# Middleware de CORS para liberar acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substitua pelo domínio do frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-cbct/")
async def upload_cbct(files: List[UploadFile] = File(...)):
    exam_id = str(uuid.uuid4())
    exam_folder = f"./uploads/{exam_id}"
    os.makedirs(exam_folder, exist_ok=True)

    for file in files:
        file_location = f"{exam_folder}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

    metadata = {
        "exam_id": exam_id,
        "data_upload": datetime.now().isoformat(),
        "arquivos": [file.filename for file in files],
        "status": "aguardando análise"
    }
    with open(f"{exam_folder}/metadados.json", "w") as meta_file:
        json.dump(metadata, meta_file, indent=4)

    return {"exam_id": exam_id}

@app.get("/ia-analisar/{exam_id}")
def analisar_cbct(exam_id: str):
    exam_folder = f"./uploads/{exam_id}"
    output_path = f"./outputs/{exam_id}"
    os.makedirs(output_path, exist_ok=True)

    result = inferir_cbct(exam_folder, output_path)

    metadata_path = f"{exam_folder}/metadados.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata["status"] = "processado"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    return {"resultado": result}

@app.get("/gerar-laudo/{exam_id}")
def gerar_laudo(exam_id: str):
    output_path = f"./outputs/{exam_id}"
    laudo_path = f"{output_path}/laudo.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Laudo Automático CBCT", ln=True, align="C")

    achados_path = f"{output_path}/achados.txt"
    if os.path.exists(achados_path):
        with open(achados_path, "r") as f:
            for linha in f:
                pdf.multi_cell(0, 10, linha.strip())
    else:
        pdf.multi_cell(0, 10, "Nenhum achado encontrado.")

    pdf.output(laudo_path)

    metadata_path = f"./uploads/{exam_id}/metadados.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata["status"] = "concluído"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    return FileResponse(laudo_path, media_type='application/pdf', filename="laudo.pdf")

@app.get("/status/{exam_id}")
def status_exame(exam_id: str):
    metadata_path = f"./uploads/{exam_id}/metadados.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return JSONResponse(content=metadata)
    return JSONResponse(content={"erro": "Exame não encontrado"}, status_code=404)
