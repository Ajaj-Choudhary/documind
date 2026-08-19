"""Splits extracted page text into overlapping, token-sized chunks for embedding."""

import re

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
except Exception:
    # tiktoken requires a one-time network download; fall back to a
    # character estimate if that's unavailable.
    _ENCODER = None
    _USE_TIKTOKEN = False

TARGET_CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60


def count_tokens(text):
    # Returns the token count of text, using tiktoken if available, otherwise a character-based estimate.
    if _USE_TIKTOKEN:
        return len(_ENCODER.encode(text))
    return max(1, len(text) // 4)


def tail_by_tokens(text, max_tokens):
    # Returns roughly the last max_tokens worth of text, used to seed overlap into the next chunk.
    if _USE_TIKTOKEN:
        tokens = _ENCODER.encode(text)
        return _ENCODER.decode(tokens[-max_tokens:])
    return text[-(max_tokens * 4):]


def split_into_paragraphs(text):
    # Splits text into paragraphs on blank-line boundaries and collapses internal whitespace.
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw_paragraphs:
        cleaned = re.sub(r"\s+", " ", p).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def split_long_paragraph(paragraph, max_tokens):
    # Splits a paragraph too large for one chunk into sentence-level pieces.
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if count_tokens(candidate) > max_tokens and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


def pack_paragraphs(paragraphs, target_tokens=TARGET_CHUNK_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS):
    # Greedily packs paragraphs into chunks under target_tokens, with overlap carried between chunks.
    units = []
    for para in paragraphs:
        if count_tokens(para) > target_tokens:
            units.extend(split_long_paragraph(para, target_tokens))
        else:
            units.append(para)

    chunks = []
    current_chunk = ""

    for unit in units:
        candidate = (current_chunk + "\n\n" + unit).strip()

        if count_tokens(candidate) > target_tokens and current_chunk:
            chunks.append(current_chunk)
            tail = tail_by_tokens(current_chunk, overlap_tokens)
            current_chunk = (tail + "\n\n" + unit).strip()
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_pages(pages, source_filename, target_tokens=TARGET_CHUNK_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS):
    # Chunks each page separately and attaches source/page/index metadata to every chunk.
    all_chunks = []
    chunk_index = 0

    for page in pages:
        paragraphs = split_into_paragraphs(page["text"])
        page_chunks = pack_paragraphs(paragraphs, target_tokens, overlap_tokens)

        for chunk_text in page_chunks:
            all_chunks.append({
                "text": chunk_text,
                "source_filename": source_filename,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "token_count": count_tokens(chunk_text),
            })
            chunk_index += 1

    return all_chunks