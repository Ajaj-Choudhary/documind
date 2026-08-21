"""Tests for retrieval.py, using real Chroma storage and a mocked embedding model."""

from unittest.mock import patch, MagicMock
import numpy as np

import app.embeddings as embeddings_module
from app.vector_store import get_collection, add_chunks
from app.retrieval import retrieve_relevant_chunks


def test_retrieve_relevant_chunks_finds_closest_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate this test's Chroma data from other runs

    collection = get_collection()
    chunks = [
        {"text": "about dogs and cats", "embedding": [9, 2, 1], "source_filename": "doc.pdf", "page_number": 1, "chunk_index": 0, "token_count": 5},
        {"text": "about the stock market", "embedding": [1, 9, 8], "source_filename": "doc.pdf", "page_number": 2, "chunk_index": 1, "token_count": 5},
    ]
    add_chunks(collection, chunks, doc_id="doc1")

    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[8, 2, 1]])

    with patch.object(embeddings_module, "get_model", return_value=fake_model):
        results = retrieve_relevant_chunks(collection, "what breed of dog?", top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "about dogs and cats"
    assert results[0]["page_number"] == 1