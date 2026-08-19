'''Vector store module for managing document chunks using Chroma.'''
import chromadb

CHROMA_PATH = "chroma_data"  # local folder where Chroma persists its data
COLLECTION_NAME = "documind_chunks"


def get_collection():
    '''Get or create the Chroma collection for storing document chunks.'''
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def add_chunk(collection, chunk, chunk_id):
    '''Add a chunk to the Chroma collection with the given chunk_id and metadata.'''
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