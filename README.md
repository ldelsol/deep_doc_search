# Deep Doc Search 📄🔍  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green)](https://faiss.ai/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)  
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)](https://streamlit.io/)  

Deep Doc Search is a document-based search and question-answering tool that leverages **vector embeddings, FAISS, and LLMs** for **fast and accurate information retrieval**.  

---

## 🚀 Features
✅ **Custom Embedding Models**: Supports `all-MiniLM-L6-v2`, `all-distilroberta-v1`, and more.  
✅ **Efficient Vector Search**: Uses **FAISS** for optimized document retrieval.  
✅ **LLM-Powered Answers**: Generates precise responses based on document content.  
✅ **Modular Architecture**: Designed for flexibility and scalability.  
✅ **Evaluation Tools**: Built-in **Recall@k**, **ROUGE-L**, and performance benchmarking.  

---

## 📖 Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)

---

## 🛠 Installation

### 1️⃣ Clone the repository  
git clone https://github.com/ldelsol/deep_doc_search.git
cd deep_doc_search

### 2️⃣ Install dependencies  
pip install -r requirements.txt

### 3️⃣ Install the package in editable mode  
pip install -e .

---

## 📌 Usage

### 🔹 1. Process and Vectorize a PDF
To extract text and generate vector embeddings:
python deep_doc_search/vector_store.py

### 🔹 2. Query the Document
Run the interactive **Streamlit** web app:
streamlit run app/app.py
This opens a UI where you can ask questions based on the document content.

### 🔹 3. Run a Query via CLI
python deep_doc_search/query_handler.py "What is the plan to protect water resources?"

---

## 📊 Evaluation

### 🔹 1. Evaluate Retrieval Performance (Recall@k)
python tests/evaluate_model.py

### 🔹 2. Test Different Embeddings and Chunk Sizes
python tests/hyperparameter_tuning.py

### 🔹 3. Measure Response Quality with ROUGE-L
python tests/test_rouge.py