from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

# Load the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Raw text inputs
text_1 = "Amazon optimizes its logistics with AI."
text_2 = "amazon optimizes its logistics with ai"  # Lowercase, no punctuation

vec1 = np.array(embeddings.embed_query(text_1))
vec2 = np.array(embeddings.embed_query(text_2))

# Compute cosine similarity between both versions
similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"Cosine similarity between the two versions: {similarity:.4f}")
