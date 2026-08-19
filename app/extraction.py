"""
extraction.py

Extracts text from uploaded files (PDF, TXT), page by page, preserving
human-readable page numbers so citations can point back to the source.
"""

import pymupdf as fitz
from pathlib import Path


class ExtractionError(Exception):
    """Raised when a file has no usable text to extract."""
    pass


def extract_pdf(file_path):
    doc = fitz.open(file_path)
    pages = []

    for i in range(len(doc)):
        text = doc[i].get_text()
        cleaned = text.strip()
        page_number = i + 1

        if cleaned:
            pages.append({"page_number": page_number, "text": cleaned})

    if not pages:
        raise ExtractionError("No extractable text found in this PDF.")

    return pages


def extract_txt(file_path):
    """
    Plain text files have no real "pages" -- we treat the whole file as
    a single logical page so the return shape matches extract_pdf(),
    keeping downstream code (chunking) agnostic to file type.
    """
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    cleaned = text.strip()

    if not cleaned:
        raise ExtractionError("Text file is empty.")

    return [{"page_number": 1, "text": cleaned}]


def extract_text(file_path, filename):
    """Single entry point -- routes to the right extractor by file extension."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(file_path)
    elif suffix == ".txt":
        return extract_txt(file_path)
    else:
        raise ExtractionError(f"Unsupported file type: {suffix}")
