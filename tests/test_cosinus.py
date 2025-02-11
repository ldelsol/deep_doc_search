from langchain_community.embeddings import HuggingFaceEmbeddings
from deep_doc_search.query_handler import normalize_query
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_1 = "Amazon optimizes its logistics with AI."
text_2 = normalize_query(text_1)

vec1 = np.array(embeddings.embed_query(text_1))
vec2 = np.array(embeddings.embed_query(text_2))

# Compute cosine similarity between both versions
similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"Cosine similarity between the two versions: {similarity:.4f}")
