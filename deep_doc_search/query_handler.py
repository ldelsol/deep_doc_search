import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import os
import re

DB_PATH = "data/faiss_index"
METADATA_PATH = "data/metadata.pkl"

def load_vector_store():
    """Loads the FAISS index and associated texts."""
    if not os.path.exists(DB_PATH) or not os.path.exists(METADATA_PATH):
        print("No FAISS database found. Run `vector_store.py` first.")
        return None, None

    index = faiss.read_index(DB_PATH)

    with open(METADATA_PATH, "rb") as f:
        documents = pickle.load(f)

    return index, documents

def normalize_query(query):
    """Normalizes the user query (lowercase, removes punctuation, trims spaces)."""
    query = query.lower() 
    query = re.sub(r'[^\w\s]', '', query)  # Remove punctuation
    query = query.strip()
    return query

def search_in_vector_store(query, k=3):
    """Searches for the `k` most similar texts to the user query."""
    index, documents = load_vector_store()
    if index is None or documents is None:
        return [], []

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-distilroberta-v1")

    query = normalize_query(query)

    query_vector = embeddings.embed_query(query)
    query_vector = np.array([query_vector]).astype('float32')

    D, I = index.search(query_vector, k)

    results = [documents[i] for i in I[0] if i != -1]
    distances = [D[0][j] for j in range(len(I[0])) if I[0][j] != -1]

    return results, distances


# Testing
if __name__ == "__main__":
    query = input("Enter your search query: ")
    results, distances = search_in_vector_store(query)

    print("\nSearch Results:")
    for i, (res, dist) in enumerate(zip(results, distances)):
        print(f"\nResult {i+1} (Distance: {dist:.4f}):\n{res}")
