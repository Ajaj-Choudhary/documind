"""Combines question embedding and vector search into one retrieval step."""

from app.embeddings import embed_texts
from app.vector_store import query_similar_chunks


def retrieve_relevant_chunks(collection, question, top_k=5):
    # Embeds the question and returns the top_k most similar stored chunks.
    question_embedding = embed_texts([question])[0]
    return query_similar_chunks(collection, question_embedding, top_k=top_k)