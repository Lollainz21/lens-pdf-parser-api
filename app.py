import os
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pdfplumber
import docx
import openai

# Legge la chiave dalle variabili di ambiente (Render)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS aperto, così puoi chiamarlo dal frontend senza problemi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
        text = "\n".join(pages_text).strip()
        if not text:
            raise ValueError("Nessun testo leggibile trovato nel PDF.")
        return text
    except Exception as e:
        raise ValueError(f"Errore durante la lettura del PDF: {e}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in document.paragraphs).strip()
        if not text:
            raise ValueError("Nessun testo leggibile trovato nel DOCX.")
        return text
    except Exception as e:
        raise ValueError(f"Errore durante la lettura del DOCX: {e}")


def summarize_with_openai(text: str) -> str:
    """
    Usa OpenAI per fare un breve riassunto del CV.
    Se qualcosa va storto, solleva un'eccezione e ci pensa l'endpoint a gestirla.
    """
    # Taglio il testo per sicurezza, così non mando cose infinite
    truncated = text[:15000]

    prompt = (
        "Sei un assistente HR. Leggi il seguente CV e produci un breve riassunto "
        "in massimo 10 bullet point, mettendo in evidenza ruolo attuale, anni di esperienza, "
        "stack tecnologico principale e soft skills più rilevanti.\n\n"
        f"CV:\n{truncated}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # se vuoi puoi cambiare modello
        messages=[
            {"role": "system", "content": "Sei un esperto HR che analizza CV."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message["content"].strip()


@app.get("/")
def root():
    return {"status": "ok", "message": "Lens CV parser API online"}


@app.post("/parse")
async def parse_cv(file: UploadFile = File(...)):
    """
    Endpoint principale:
    - accetta PDF o DOCX
    - estrae il testo
    - prova a generare un riassunto con OpenAI
    - restituisce sempre il testo; il riassunto può essere None se OpenAI fallisce
    """
    filename = file.filename.lower()

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File vuoto.")

    # Parse del file
    try:
        if filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_bytes)
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            extracted_text = extract_text_from_docx(file_bytes)
        else:
            raise HTTPException(
                status_code=400,
                detail="Formato non supportato. Carica un file PDF o DOCX.",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=500, detail="Errore imprevisto durante il parsing del file."
        )

    # Prova a riassumere con OpenAI, ma senza bloccare tutto se fallisce
    summary = None
    try:
        summary = summarize_with_openai(extracted_text)
    except Exception as e:
        # Qui potresti loggare e basta; non buttiamo giù l'API per colpa di OpenAI
        print(f"Errore durante il riassunto con OpenAI: {e}")

    return {
        "filename": file.filename,
        "text": extracted_text,
        "summary": summary,
    }
