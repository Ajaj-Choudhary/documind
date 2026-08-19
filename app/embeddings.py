"""
embeddings.py

Turns chunk text into vectors using OpenAI's embedding API.

Model choice: text-embedding-3-small
- 1536 dimensions
- Cheap and fast, strong quality-to-cost ratio for a portfolio project
- Easy to justify in interviews: "small" model is the right call when
  you don't have millions of documents and cost/latency matter more
  than squeezing out the last bit of retrieval quality
"""

import os
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

_client = None


def get_client():
    """
    Lazily create the OpenAI client so importing this module doesn't
    fail just because OPENAI_API_KEY isn't set yet (e.g. during tests
    that don't need real API calls).
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before calling embedding functions."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def embed_texts(texts):
    """
    Embeds a list of strings in a single API call (much more efficient
    than one call per chunk -- fewer network round trips, and the API
    supports batching natively).

    Returns a list of vectors (list[float]), same order as input texts.
    """
    if not texts:
        return []

    client = get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # response.data is not guaranteed to preserve input order across all
    # client versions -- sort by the "index" field to be safe rather
    # than assuming order, since a silently-misaligned embedding would
    # be a very hard bug to notice later.
    sorted_data = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in sorted_data]


def embed_chunks(chunks):
    """
    Takes chunk dicts from chunk_pages() and returns the same dicts
    with an "embedding" field added to each one.
    """
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks
