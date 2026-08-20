"""Tests for embeddings.py, using a mocked SentenceTransformer model (no real download/inference)."""

from unittest.mock import patch, MagicMock
import numpy as np

import app.embeddings as embeddings_module
from app.embeddings import embed_texts, embed_chunks


def test_embed_texts_returns_plain_lists():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    with patch.object(embeddings_module, "get_model", return_value=fake_model):
        result = embed_texts(["first", "second"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_chunks_preserves_metadata():
    chunks = [
        {"text": "first", "source_filename": "a.txt", "page_number": 1, "chunk_index": 0, "token_count": 5},
        {"text": "second", "source_filename": "a.txt", "page_number": 1, "chunk_index": 1, "token_count": 5},
    ]
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    with patch.object(embeddings_module, "get_model", return_value=fake_model):
        result = embed_chunks(chunks)

    assert result[0]["embedding"] == [0.1, 0.2]
    assert result[0]["source_filename"] == "a.txt"
    assert result[1]["embedding"] == [0.3, 0.4]


def test_embed_texts_empty_list_skips_model_call():
    with patch.object(embeddings_module, "get_model") as mock_get_model:
        result = embed_texts([])

    assert result == []
    mock_get_model.assert_not_called()