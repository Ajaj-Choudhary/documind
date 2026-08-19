"""
main.py

FastAPI layer. Accepts an uploaded file, runs it through
extraction -> chunking, and returns the result.

Not yet included: embeddings, vector storage, auth. Those land in
later modules and will replace the "return chunks directly" behavior
below with "store chunks, return a doc_id."
"""

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.extraction import extract_text, ExtractionError
from app.chunking import chunk_pages

app = FastAPI(title="DocuMind API")

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    doc_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{doc_id}{suffix}"

    with saved_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    try:
        pages = extract_text(str(saved_path), file.filename)
        chunks = chunk_pages(pages, source_filename=file.filename)
    except ExtractionError as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
