"""Calls the Claude API to generate an answer grounded in retrieved chunks."""

import os
from anthropic import Anthropic

from app.prompting import SYSTEM_PROMPT, build_prompt

MODEL_NAME = "claude-sonnet-4-6"

_client = None


def get_client():
    # Lazily creates and caches the Anthropic client from the API key in the environment.
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
        _client = Anthropic(api_key=api_key)
    return _client


def generate_answer(question, chunks):
    # Builds a grounded prompt from the retrieved chunks and asks Claude to answer it.
    if not chunks:
        return "No relevant content was found in your documents to answer this question."

    prompt = build_prompt(question, chunks)
    client = get_client()

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text