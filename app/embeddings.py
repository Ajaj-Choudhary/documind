"""Embeds chunk text using OpenAI's text-embedding-3-small (1536 dimensions)."""

import os
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

_client = None


def get_client():
    # Lazily creates and caches the OpenAI client from the API key in the environment.
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def embed_texts(texts):
    # Embeds a batch of strings in one API call, returned in the same order as input.
    if not texts:
        return []

    client = get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)

    # Re-sort by index rather than trusting response order, to avoid
    # silently misaligning an embedding with the wrong chunk.
    sorted_data = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in sorted_data]


def embed_chunks(chunks):
    # Embeds each chunk's text and attaches the resulting vector to its dict.
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks