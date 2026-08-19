"""Wraps Chroma for storing and retrieving embedded document chunks."""

import chromadb

CHROMA_PATH = "chroma_data"
COLLECTION_NAME = "documind_chunks"


def get_collection():
    # Connects to the local Chroma database and returns the chunks collection.
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def add_chunk(collection, chunk, chunk_id):
    # Stores a single embedded chunk with its text and metadata under a unique id.
    collection.add(
        ids=[chunk_id],
        embeddings=[chunk["embedding"]],
        documents=[chunk["text"]],
        metadatas=[{
            "source_filename": chunk["source_filename"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "token_count": chunk["token_count"],
        }],
    )