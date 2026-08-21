"""FastAPI layer: accepts an uploaded file, runs extraction, chunking, embedding, and storage."""

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.extraction import extract_text, ExtractionError
from app.chunking import chunk_pages
from app.embeddings import embed_chunks
from app.vector_store import get_collection, add_chunks
from app.retrieval import retrieve_relevant_chunks
from app.generation import generate_answer

app = FastAPI(title="DocuMind API")

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}

collection = get_collection()


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    # Saves the uploaded file, extracts/chunks/embeds its text, and stores it in Chroma.
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

    chunks = embed_chunks(chunks)
    add_chunks(collection, chunks, doc_id=doc_id)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
    }


@app.post("/documents/ask")
async def ask_question(question: str):
    # Retrieves relevant chunks and asks Claude to generate a grounded, cited answer.
    chunks = retrieve_relevant_chunks(collection, question, top_k=5)
    answer = generate_answer(question, chunks)
    return {"question": question, "answer": answer, "sources": chunks}


@app.get("/health")
async def health():
    # Simple liveness check.
    return {"status": "ok"}