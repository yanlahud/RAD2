from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import uuid
from ia_infer import inferir_cbct

app = FastAPI()

# CORS configurado com domínio do frontend Vite
origins = [
    "https://vitejsvitep6cu88rd-qmor--5173--4d9fd228.local-credentialless.webcontainer.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-cbct/")
async def upload_cbct(file: UploadFile = File(...)):
    try:
        exam_id = str(uuid.uuid4())
        upload_dir = f"./uploads/{exam_id}"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"exam_id": exam_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ia-analisar/{exam_id}")
def ia_analisar(exam_id: str):
    try:
        input_dir = f"./uploads/{exam_id}"
        output_dir = f"./outputs/{exam_id}"
        os.makedirs(output_dir, exist_ok=True)
        result = inferir_cbct(input_dir, output_dir)
        return {"resultado": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gerar-laudo/{exam_id}")
def gerar_laudo(exam_id: str):
    try:
        laudo_path = f"./outputs/{exam_id}/laudo.pdf"
        if not os.path.exists(laudo_path):
            return {"erro": "Laudo ainda não gerado."}
        return FileResponse(laudo_path, media_type='application/pdf', filename="laudo.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
