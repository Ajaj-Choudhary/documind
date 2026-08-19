"""Extracts text from uploaded PDF and TXT files, page by page."""

import pymupdf as fitz
from pathlib import Path


class ExtractionError(Exception):
    pass


def extract_pdf(file_path):
    # Extracts text page by page from a PDF, skipping blank pages.
    doc = fitz.open(file_path)
    pages = []

    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        if text:
            pages.append({"page_number": i + 1, "text": text})

    if not pages:
        raise ExtractionError("No extractable text found in this PDF.")

    return pages


def extract_txt(file_path):
    # Reads a plain text file, returning it as a single logical page.
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ExtractionError("Text file is empty.")
    return [{"page_number": 1, "text": text}]


def extract_text(file_path, filename):
    # Routes to the correct extractor based on file extension.
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(file_path)
    elif suffix == ".txt":
        return extract_txt(file_path)
    else:
        raise ExtractionError(f"Unsupported file type: {suffix}")