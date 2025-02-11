import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from deep_doc_search.pdf_processing import extract_text_from_pdf
import os
import pickle 

DB_PATH = "data/faiss_index"
METADATA_PATH = "data/metadata.pkl" 

def create_vector_store_from_pdf(pdf_path):
    """Converts text from a PDF into embeddings and stores them in FAISS"""

    text = extract_text_from_pdf(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectors = [embeddings.embed_query(doc) for doc in documents]
    vectors = np.array(vectors).astype('float32')

    dimension = vectors.shape[1]  # Embedding size
    index = faiss.IndexFlatL2(dimension) 
    index.add(vectors)

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
