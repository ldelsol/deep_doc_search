import time
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from deep_doc_search.query_handler import search_in_vector_store
from deep_doc_search.llm_handler import generate_response

# List of questions to test performance
queries = [
    "What are LVMH’s sustainability goals?",
    "Who is the CEO of LVMH?",
    "What was LVMH’s revenue in 2023?",
    "How does LVMH invest in innovation?",
    "What is LVMH’s strategy for digital transformation?",
]

def measure_faiss_search_time(query, k=3):
    """Measures the time required to retrieve a response via FAISS."""
    start_time = time.time()
    results, _ = search_in_vector_store(query, k)
    elapsed_time = time.time() - start_time
    return elapsed_time, results

def measure_llm_response_time(query, context):
    """Measures the time required to generate a response via the LLM."""
    start_time = time.time()
    
    prompt = f"""
    You are an AI assistant specialized in analyzing internal documents.
    Here is an excerpt from the document that may help answer the question:

    {context}

    Question: {query}
    Respond accurately and concisely using only the provided information.
    """
    response = generate_response(prompt)
    
    elapsed_time = time.time() - start_time
    return elapsed_time, response

def test_performance(queries, k=3):
    """Tests the speed of FAISS and the LLM for multiple queries."""
    faiss_times = []
    llm_times = []
    total_times = []

    print("\n🔍 PERFORMANCE TEST: FAISS & LLM\n")

    for query in queries:
        print(f"🟢 Query: {query}")

        # Measure FAISS search time
        faiss_time, results = measure_faiss_search_time(query, k)
        faiss_times.append(faiss_time)
        print(f"   ⏳ FAISS Search Time: {faiss_time:.4f} sec")

        # Concatenate FAISS results as context for the LLM
        context = "\n\n".join(results)

        # Measure LLM response time
        llm_time, _ = measure_llm_response_time(query, context)
        llm_times.append(llm_time)
        print(f"   🤖 LLM Response Time: {llm_time:.4f} sec")

        # Total FAISS + LLM time
        total_time = faiss_time + llm_time
        total_times.append(total_time)
        print(f"   🏁 Total Response Time: {total_time:.4f} sec\n")

    # Display average times
    avg_faiss_time = np.mean(faiss_times)
    avg_llm_time = np.mean(llm_times)
    avg_total_time = np.mean(total_times)

    print("\n📊 OVERALL RESULTS:")
    print(f"✅ Average FAISS search time: {avg_faiss_time:.4f} sec")
    print(f"✅ Average LLM response time: {avg_llm_time:.4f} sec")
    print(f"✅ Average total time (FAISS + LLM): {avg_total_time:.4f} sec")

if __name__ == "__main__":
    test_performance(queries, k=3)
