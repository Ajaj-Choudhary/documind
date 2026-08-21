"""Tests for generation.py, using a mocked Anthropic client (no real API calls)."""

from unittest.mock import patch, MagicMock

import app.generation as generation_module
from app.generation import generate_answer


def test_generate_answer_returns_claude_response_text():
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="Answer with a citation [notes.pdf, page 3].")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    chunks = [{"text": "some content", "source_filename": "notes.pdf", "page_number": 3}]

    with patch.object(generation_module, "get_client", return_value=mock_client):
        answer = generate_answer("What is RAG?", chunks)

    assert answer == "Answer with a citation [notes.pdf, page 3]."


def test_generate_answer_skips_api_call_when_no_chunks():
    mock_client = MagicMock()

    with patch.object(generation_module, "get_client", return_value=mock_client):
        answer = generate_answer("What is RAG?", [])

    assert "No relevant content" in answer
    mock_client.messages.create.assert_not_called()