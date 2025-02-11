import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deep_doc_search.query_handler import normalize_query
from deep_doc_search.pdf_processing import extract_text_from_pdf

EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-distilroberta-v1",
]

# List of chunk sizes and overlaps to test
CHUNK_SIZES = [200, 500, 1000]
CHUNK_OVERLAPS = [50, 100]

QUERY = "What is the plan to protect water resources?"
GROUND_TRUTH = "The goal is a 30% reduction in the amount of water used by LVMH’s operations and its value chain by 2030"

def normalize_text(text):
    """Cleans text by removing extra spaces, line breaks, and converting to lowercase."""
    return " ".join(text.lower().strip().split())

def evaluate_embedding_quality(model_name, chunk_size, chunk_overlap, text):
    """Tests an embedding model with given chunk size and overlap, evaluates FAISS retrieval."""
    print(f"\nTesting {model_name} | Chunk Size: {chunk_size} | Chunk Overlap: {chunk_overlap}")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)

    vectors = [embeddings.embed_query(chunk) for chunk in chunks]
    vectors = np.array(vectors, dtype="float32")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    query_vector = np.array([embeddings.embed_query(QUERY)], dtype="float32")
    D, I = index.search(query_vector, k=5)  # Retrieve up to 5 results

    found_rank = -1
    best_distance = float('inf')

    print(f"\nResults for Query: {QUERY}")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        chunk_text = chunks[idx]
        normalized_chunk = normalize_query(chunk_text)
        normalized_ground_truth = normalize_query(GROUND_TRUTH)

        print(f"📊 Rank {rank + 1} | Distance: {dist:.4f} | Chunk: {chunk_text}...")

        if normalized_ground_truth in normalized_chunk:
            found_rank = rank + 1
            best_distance = dist
            break  

    if found_rank != -1:
        print(f"\nGround truth found at rank {found_rank} with a distance of {best_distance:.4f}")
    else:
        print("\nGround truth not found in the top 5 results.")

    return found_rank, best_distance

pdf_path = "data/test.pdf"
text = extract_text_from_pdf(pdf_path)

# Test all hyperparameter combinations
results = []
for model in EMBEDDING_MODELS:
    for chunk_size in CHUNK_SIZES:
        for chunk_overlap in CHUNK_OVERLAPS:
            found_rank, best_distance = evaluate_embedding_quality(model, chunk_size, chunk_overlap, text)
            results.append((model, chunk_size, chunk_overlap, found_rank, best_distance))

print("\nSUMMARY OF TEST RESULTS:")
for model, chunk_size, chunk_overlap, rank, dist in results:
    if rank != -1:
        print(f"Model: {model} | Chunk Size: {chunk_size} | Overlap: {chunk_overlap} | Found at Rank: {rank} | Distance: {dist:.4f}")
    else:
        print(f"Model: {model} | Chunk Size: {chunk_size} | Overlap: {chunk_overlap} | Ground truth NOT FOUND in the top 5")
