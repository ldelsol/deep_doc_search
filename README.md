# Deep Doc Search

Deep Doc Search is a powerful tool for document search and question-answering using embeddings and vector similarity. It supports PDF ingestion, vectorization, and natural language queries.

## Features
- **Custom Embedding Models**: Support for various embedding models like `all-MiniLM-L6-v2` and `all-distilroberta-v1`.
- **Chunk-Based Vectorization**: Flexible chunk sizes to optimize retrieval performance.
- **FAISS Integration**: Efficient vector search for quick and accurate results.
- **LLM Integration**: Precise answers generated using language models.
- **Evaluation Tools**: Includes Recall@k, ROUGE-L scoring, and performance benchmarking.

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed. Create a virtual environment for better dependency management:
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

### Steps to Install
1. Clone the repository:
   git clone https://github.com/ldelsol/deep_doc_search.git
   cd deep_doc_search

2. Install dependencies:
   pip install -r requirements.txt

3. Install the project in editable mode:
   pip install -e .

## Usage

### 1. Ingest and Vectorize a PDF
To process a PDF and create a vector store, run the following:
python deep_doc_search/vector_store.py
This will create a FAISS vector index from the content of the PDF.

### 2. Query the Document
Launch the Streamlit application to query your document:
streamlit run app/app.py
This opens an interactive interface for querying and retrieving answers.

### 3. Evaluate the Model
Run the evaluation script to test Recall@k and ROUGE-L metrics:
python tests/evaluate_model.py

### 4. Test Hyperparameters
You can test different embedding models and chunk sizes:
python tests/hyperparameter_tuning.py

## Directory Structure
deep_doc_search/
│
├── app/                     # Streamlit application
├── data/                    # Data files (PDFs, vector store)
├── deep_doc_search/         # Core package
│   ├── llm_handler.py       # LLM integration
│   ├── pdf_processing.py    # PDF processing logic
│   ├── query_handler.py     # Query and FAISS interaction
│   └── vector_store.py      # Vector store creation
├── tests/                   # Evaluation and performance tests
├── README.md                # Project overview
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── .gitignore               # Files to ignore in version control