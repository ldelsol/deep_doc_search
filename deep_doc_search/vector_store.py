import faiss  # Optimized for similarity search between vectors
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pdf_processing import extract_text_from_pdf
import os
import pickle  # For saving FAISS index to disk

# Storage file paths
DB_PATH = "data/faiss_index"
METADATA_PATH = "data/metadata.pkl"  # Stores text associated with embeddings

def create_vector_store_from_pdf(pdf_path):
    """Converts text from a PDF into embeddings and stores them in FAISS"""

    # Step 1: Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_text(text)

    # Step 3: Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-distilroberta-v1")

    # Step 4: Generate embeddings
    vectors = [embeddings.embed_query(doc) for doc in documents]
    vectors = np.array(vectors).astype('float32')

    # Step 5: Create the FAISS index
    dimension = vectors.shape[1]  # Embedding size (e.g., 384)
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity search
    index.add(vectors)  # Add embeddings to FAISS index

    # Step 6: Save FAISS index and associated text
    faiss.write_index(index, DB_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("FAISS vector database successfully created!")
    return index, documents

# Execute script to create vector store
if __name__ == "__main__":
    pdf_path = "data/test.pdf"

    if not os.path.exists(DB_PATH):
        create_vector_store_from_pdf(pdf_path)
    else:
        print("FAISS vector database already exists.")
