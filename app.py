from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx
import tempfile
import openai
import os
import json
from typing import Dict

app = FastAPI()

# Allow Lovable origin or '*' for testing (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST","OPTIONS","GET"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")

openai.api_key = OPENAI_API_KEY

def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def clean_text(t: str) -> str:
    # semplice pulizia, puoi migliorare
    return " ".join(t.split())

async def call_openai_extract_structured(text: str) -> Dict:
    # Prompt per OpenAI per restituire JSON strutturato
    system = "You are an expert HR parser. Return valid JSON with fields: name, contact, summary, hard_skills[], soft_skills[], experience[{company, role, start, end, desc}], education[], certifications[], languages[], raw_text_preview (first 400 chars)."
    user = f"Extract structured CV data from the text below. Text:\n\n{text}"

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # o un modello che hai accesso
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        temperature=0.0,
        max_tokens=1000,
    )
    content = resp.choices[0].message["content"]
    # Try to parse JSON out of the response
    try:
        # in molti casi GPT restituisce JSON direttamente
        parsed = json.loads(content)
        return parsed
    except Exception:
        # Se non è JSON pulito, tenta di estrarre l'ultima porzione JSON
        import re
        m = re.search(r'(\{.*\})', content, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass
        # fallback: restituisci testo grezzo in payload
        return {"error":"could_not_parse_json","raw_output": content, "raw_text_preview": text[:400]}

@app.post("/parse")
async def parse_cv(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = file.filename.lower().split('.')[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()

    try:
        if suffix == "pdf":
            raw = extract_text_from_pdf(tmp.name)
        elif suffix in ("docx","doc"):
            raw = extract_text_from_docx(tmp.name)
        elif suffix == "txt":
            raw = content.decode('utf-8', errors='ignore')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    cleaned = clean_text(raw)
    if len(cleaned.strip()) < 20:
        raise HTTPException(status_code=400, detail="Extracted text too short or empty")

    # call OpenAI to structure the CV
    structured = await call_openai_extract_structured(cleaned)

    return {"filename": file.filename, "text": cleaned, "structured": structured}
