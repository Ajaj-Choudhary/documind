"""Embeds chunk text using a local sentence-transformers model (no API key or cost)."""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

_model = None


def get_model():
    # Lazily loads and caches the local embedding model.
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    # Embeds a batch of strings locally, returned in the same order as input.
    if not texts:
        return []

    model = get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return [vector.tolist() for vector in vectors]


def embed_chunks(chunks):
    # Embeds each chunk's text and attaches the resulting vector to its dict.
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks