"""Tests for embeddings.py, using a mocked OpenAI client (no real API calls)."""

import os
from unittest.mock import patch, MagicMock

import pytest

import app.embeddings as embeddings_module
from app.embeddings import embed_texts, embed_chunks, get_client


def test_missing_api_key_raises_clear_error():
    # Test that get_client raises a RuntimeError with a clear message when the OPENAI_API_KEY environment variable is not set.
    os.environ.pop("OPENAI_API_KEY", None)
    embeddings_module._client = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_client()


def test_embed_texts_sorts_by_index():
    # Test that embed_texts returns embeddings sorted by the original input order, even if the API response is out of order.
    fake_response = MagicMock()
    fake_response.data = [
        MagicMock(index=2, embedding=[0.3, 0.3]),
        MagicMock(index=0, embedding=[0.1, 0.1]),
        MagicMock(index=1, embedding=[0.2, 0.2]),
    ]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = fake_response

    with patch.object(embeddings_module, "get_client", return_value=mock_client):
        result = embed_texts(["first", "second", "third"])

    assert result == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]


def test_embed_chunks_preserves_metadata():
    # Test that embed_chunks adds embeddings to each chunk while preserving the original metadata.
    chunks = [
        {"text": "first", "source_filename": "a.txt", "page_number": 1, "chunk_index": 0, "token_count": 5},
        {"text": "second", "source_filename": "a.txt", "page_number": 1, "chunk_index": 1, "token_count": 5},
    ]
    fake_response = MagicMock()
    fake_response.data = [
        MagicMock(index=0, embedding=[0.1, 0.1]),
        MagicMock(index=1, embedding=[0.2, 0.2]),
    ]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = fake_response

    with patch.object(embeddings_module, "get_client", return_value=mock_client):
        result = embed_chunks(chunks)

    assert result[0]["embedding"] == [0.1, 0.1]
    assert result[0]["source_filename"] == "a.txt"
    assert result[1]["embedding"] == [0.2, 0.2]


def test_embed_texts_empty_list_skips_api_call():
    # Test that embed_texts returns an empty list and does not call the OpenAI API when given an empty input list.
    with patch.object(embeddings_module, "get_client") as mock_get_client:
        result = embed_texts([])

    assert result == []
    mock_get_client.assert_not_called()