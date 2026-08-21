"""Tests for prompting.py."""

from app.prompting import build_prompt


def test_build_prompt_includes_source_and_page_for_each_chunk():
    chunks = [
        {"text": "first excerpt", "source_filename": "a.pdf", "page_number": 1},
        {"text": "second excerpt", "source_filename": "a.pdf", "page_number": 2},
    ]

    prompt = build_prompt("a question", chunks)

    assert "[a.pdf, page 1]" in prompt
    assert "first excerpt" in prompt
    assert "[a.pdf, page 2]" in prompt
    assert "second excerpt" in prompt
    assert "Question: a question" in prompt