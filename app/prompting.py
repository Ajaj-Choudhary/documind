"""Builds the prompt sent to Claude, grounding answers in retrieved chunks."""

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided document excerpts.

Rules:
- Only use information from the excerpts below to answer.
- If the excerpts don't contain enough information to answer, say so directly.
- After each claim, cite the source using this exact format: [source_filename, page page_number].
- Do not make up information that isn't in the excerpts."""


def build_prompt(question, chunks):
    # Formats retrieved chunks and the question into a single user message for Claude.
    excerpt_blocks = []
    for chunk in chunks:
        excerpt_blocks.append(
            f"[{chunk['source_filename']}, page {chunk['page_number']}]\n{chunk['text']}"
        )
    excerpts_text = "\n\n---\n\n".join(excerpt_blocks)

    return f"""Document excerpts:

{excerpts_text}

Question: {question}"""