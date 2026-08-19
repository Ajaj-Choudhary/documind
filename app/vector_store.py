"""
vector_store.py

Day 1: just connect to Chroma and get (or create) a collection.
A "collection" in Chroma is like a table -- a named group of vectors.
We're not storing or searching anything yet, just proving the
connection works.
"""

import chromadb

CHROMA_PATH = "chroma_data"  # local folder where Chroma persists its data
COLLECTION_NAME = "documind_chunks"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection
