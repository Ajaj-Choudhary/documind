"""
chunking.py

Splits extracted page text into chunks suitable for embedding.
Currently implements paragraph splitting; packing paragraphs into
properly-sized chunks (with overlap) is still in progress.
"""

import re


def split_into_paragraphs(text):
    """
    Splits text into paragraphs on blank-line boundaries (a newline,
    optional stray whitespace, then another newline), and collapses
    any mid-paragraph line breaks left over from PDF text wrapping.
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)

    paragraphs = []
    for p in raw_paragraphs:
        collapsed = re.sub(r"\s+", " ", p)
        stripped = collapsed.strip()
        if stripped:
            paragraphs.append(stripped)

    return paragraphs
