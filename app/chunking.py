"""
chunking.py

Splits extracted page text into overlapping, token-sized chunks ready
for embedding, with metadata (source file, page, chunk index) attached
to each one so citations can point back to the original document.
"""

import re

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
except Exception:
    # tiktoken needs a one-time network download of its vocab file.
    # If that's unavailable (offline/restricted network), fall back to
    # a rough character-based estimate instead of crashing.
    _ENCODER = None
    _USE_TIKTOKEN = False

TARGET_CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60


def count_tokens(text):
    if _USE_TIKTOKEN:
        return len(_ENCODER.encode(text))
    return max(1, len(text) // 4)


def tail_by_tokens(text, max_tokens):
    if _USE_TIKTOKEN:
        tokens = _ENCODER.encode(text)
        return _ENCODER.decode(tokens[-max_tokens:])
    approx_chars = max_tokens * 4
    return text[-approx_chars:]


def split_into_paragraphs(text):
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw_paragraphs:
        collapsed = re.sub(r"\s+", " ", p)
        stripped = collapsed.strip()
        if stripped:
            paragraphs.append(stripped)
    return paragraphs


def split_long_paragraph(paragraph, max_tokens):
    """
    Fallback for a single paragraph that's already bigger than our
    target chunk size on its own (common in dense text with no natural
    paragraph breaks). Splits by sentence instead of raw characters, so
    we still avoid cutting mid-sentence.
    """
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
    """
    Greedily packs paragraphs into chunks under target_tokens, carrying
    a small overlap from the end of each sealed chunk into the start of
    the next one so context isn't lost at chunk boundaries.
    """
    # Expand any individually-oversized paragraph before packing.
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
    """
    Main entry point. Takes extracted pages (from extraction.py) and
    returns chunk dicts ready for embedding:

        {
            "text": "...",
            "source_filename": "paper.pdf",
            "page_number": 3,
            "chunk_index": 7,
            "token_count": 384,
        }

    Runs pack_paragraphs() once PER PAGE (not on the whole document at
    once) -- this is what lets each chunk keep an accurate page_number,
    since that information only exists at the page level.
    """
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
