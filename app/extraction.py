"""
extraction.py

Extracts text from uploaded PDF files, page by page, preserving each
page's human-readable page number (not the 0-indexed internal one) so
that later features (citations) can point back to the correct page.
"""

import pymupdf as fitz


class ExtractionError(Exception):
    """Raised when a file has no usable text to extract."""
    pass


def extract_pdf(file_path):
    doc = fitz.open(file_path)
    pages = []

    for i in range(len(doc)):
        text = doc[i].get_text()
        cleaned = text.strip()
        page_number = i + 1  # convert 0-indexed -> human page number

        if cleaned:  # skip blank/image-only pages
            pages.append({"page_number": page_number, "text": cleaned})

    if not pages:
        # Fails loudly instead of silently returning an empty list --
        # an empty list here would quietly propagate into chunking and
        # produce a "successful" upload with zero usable content.
        raise ExtractionError("No extractable text found in this PDF.")

    return pages
