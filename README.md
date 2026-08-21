# DocuMind

An AI-powered document Q&A assistant. Upload PDFs or text files, then ask questions and get answers grounded in the actual document content, with citations back to the source page.

## How it works

```
upload → extract text (per page) → chunk (paragraph-aware, with overlap)
       → embed → store in vector DB
                                        ↓
question → embed → retrieve top-k similar chunks → Claude generates
                                                     a cited answer
```

## Stack

- **Backend:** Python, FastAPI
- **Text extraction:** PyMuPDF (PDF), built-in file reading (TXT)
- **Chunking:** paragraph-aware, token-sized, with overlap between chunks
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`), run locally — no API cost
- **Vector DB:** Chroma (local, persistent)
- **LLM:** Claude (Anthropic API), for answer generation and citations

## Why paragraph-aware chunking, not fixed-size

Splitting text every N characters cuts sentences and paragraphs in half, which lowers embedding quality — an embedding of a half-formed thought is a worse "fingerprint" of its meaning than a whole one. This project instead splits on natural paragraph boundaries first, and only falls back to sentence-level splitting for individual paragraphs too large to fit in one chunk on their own.

Each chunk also carries a small overlap (roughly 15% of its size) from the tail of the previous chunk, so an idea that spans a chunk boundary isn't lost. Without this, a sentence like "It reduced errors by 40%" can end up separated from the earlier sentence explaining what "it" refers to.

## Why local embeddings instead of a paid API

`sentence-transformers` runs entirely on-device after a one-time model download, with no per-request cost and no API key required. This kept development and testing free while iterating quickly, and is a reasonable production choice too when cost or data-residency matter more than squeezing out the last bit of retrieval quality that a larger hosted embedding model might offer.

## Known limitations

- A single paragraph that is one long sentence (no internal punctuation) can't be split further by the sentence-level fallback, and may produce a chunk larger than the target size.
- Overlap is computed from a rough character-based estimate when `tiktoken`'s vocabulary file can't be downloaded (e.g. restricted network environments); on a machine with normal internet access this automatically uses real token boundaries instead.
- No OCR support — scanned PDFs (images with no embedded text layer) will fail extraction with a clear error rather than silently returning nothing.

## Running locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Set `ANTHROPIC_API_KEY` in your environment before calling `/documents/ask` — answer generation requires it.

```bash
curl -X POST http://localhost:8000/documents/upload -F "file=@yourfile.pdf"
curl -X POST "http://localhost:8000/documents/ask?question=your+question+here"
```

## Tests

```bash
pytest tests/ -v
```

All tests mock external calls (embedding model, Claude API) so the suite runs without any API key or network access.

## Status

Core backend pipeline (extraction, chunking, embedding, storage, retrieval, cited answer generation) is complete and tested. Frontend, authentication, and chat history persistence are still in progress.