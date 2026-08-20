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


def add_chunks(collection, chunks, doc_id):
    # Stores many embedded chunks in a single batched call, id'd by doc_id + chunk_index.
    ids = [f"{doc_id}-{chunk['chunk_index']}" for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{
        "source_filename": chunk["source_filename"],
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"],
        "token_count": chunk["token_count"],
    } for chunk in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_similar_chunks(collection, query_embedding, top_k=5):
    # Returns the top_k stored chunks most similar to the given query embedding.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": text,
            "source_filename": metadata["source_filename"],
            "page_number": metadata["page_number"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance,
        })

    return chunks