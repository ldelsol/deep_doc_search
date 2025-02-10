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

    query_vector = embeddings.embed_query(query)
    query_vector = np.array([query_vector]).astype('float32')

    D, I = index.search(query_vector, k)

    results = [documents[i] for i in I[0] if i != -1]
    distances = [D[0][j] for j in range(len(I[0])) if I[0][j] != -1]

    return results, distances

# Tested with: "WHAT ARE LVMH’S SUSTAINABILITY GOALS?"
if __name__ == "__main__":
    query = input("Enter your search query: ")

    # Search without normalization
    results_raw, distances_raw = search_in_vector_store(query)

    # Search with normalization
    query_normalized = normalize_query(query)
    results_norm, distances_norm = search_in_vector_store(query_normalized)

    print("\nResults WITHOUT normalization:")
    for i, (res, dist) in enumerate(zip(results_raw, distances_raw)):
        print(f"\nResult {i+1} (Distance: {dist:.4f}):\n{res}")

    print("\nResults WITH normalization:")
    for i, (res, dist) in enumerate(zip(results_norm, distances_norm)):
        print(f"\nResult {i+1} (Distance: {dist:.4f}):\n{res}")

    print("\nComparison of results:")
    if results_raw == results_norm:
        print("Normalization does not change the results.")
    else:
        print("Normalization has modified the FAISS results!")
