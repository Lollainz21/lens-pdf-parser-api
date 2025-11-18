from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pdfplumber
import io
from pydantic import BaseModel

app = FastAPI(title="Lens CV Parser API", version="1.0.0")

# Modello per il dato di output strutturato
class CVData(BaseModel):
    summary: str = "" # Useremo questo campo per restituire il testo completo

@app.post("/parse", response_model=CVData)
async def parse_cv(file: UploadFile):
    """
    Endpoint per ricevere un file PDF e restituire il testo estratto.
    """
    if file.content_type != 'application/pdf' and file.content_type != 'application/octet-stream':
        raise HTTPException(status_code=400, detail="Only PDF files are supported for parsing.")

    try:
        file_bytes = await file.read()

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""

        if not text.strip():
            raise HTTPException(status_code=500, detail="Could not extract any readable text from the PDF file.")

        # Ritorna il testo completo nel campo 'summary'.
        return CVData(summary=text.strip())

    except Exception as e:
        print(f"Parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")
