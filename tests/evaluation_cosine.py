import json
import numpy as np
import re
from rouge_score import rouge_scorer
from langchain_huggingface import HuggingFaceEmbeddings
from deep_doc_search.query_handler import search_in_vector_store
from deep_doc_search.llm_handler import generate_response

# Load evaluation queries and ground truths from JSON file
def load_evaluation_data():
    """Loads evaluation queries and ground truths from a JSON file."""
    with open("data/evaluation_data.json", "r") as f:
        data = json.load(f)
    return data["queries"], data["ground_truths"]

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def generate_contextual_response(query):
    """Generates an LLM response using context retrieved from FAISS, mimicking the Streamlit app."""
    search_results, _ = search_in_vector_store(query, k=3)  

    if search_results:
        context = "\n\n".join(search_results)  

        prompt = f"""
        You are an AI assistant specialized in analyzing internal documents.
        Here is an excerpt from the document that may help answer the question:

        {context}

        Question: {query}
        Respond accurately and concisely using only the provided information.
        """

        return generate_response(prompt)
    else:
        return "No relevant information found in the document."

def extract_most_relevant_sentence(generated_response, expected_answer):
    """Finds the most relevant sentence in the generated response that is closest to the expected answer."""
    sentences = re.split(r'(?<=[.!?]) +', generated_response)  # Split response into sentences
    expected_vector = np.array(embeddings.embed_query(expected_answer), dtype="float32")

    best_sentence = None
    best_similarity = -1

    for sentence in sentences:
        sentence_vector = np.array(embeddings.embed_query(sentence), dtype="float32")
        similarity = np.dot(expected_vector, sentence_vector) / (np.linalg.norm(expected_vector) * np.linalg.norm(sentence_vector))

        if similarity > best_similarity:
            best_similarity = similarity
            best_sentence = sentence

    return best_sentence, best_similarity

def evaluate_similarity_cosine(queries, ground_truths):
    """Automatically validates correctness of LLM responses using cosine similarity on the most relevant sentence."""
    scores = []

    for query, expected_answer in zip(queries, ground_truths):
        generated_response = generate_contextual_response(query)  # Generates response like in the Streamlit app

        # Find the most relevant sentence in the generated response
        best_sentence, similarity = extract_most_relevant_sentence(generated_response, expected_answer)
        scores.append(similarity)

        print(f"\n🔍 Query: {query}")
        print(f"   ✅ Expected Answer: {expected_answer}")
        print(f"   🤖 Generated Response: {generated_response}")
        print(f"   🎯 Most Relevant Sentence: {best_sentence}")
        print(f"   📊 Adjusted Cosine Similarity: {similarity:.4f}")

    avg_similarity = np.mean(scores)
    print(f"\n📊 Average Adjusted Cosine Similarity Score: {avg_similarity:.2f}")
    return avg_similarity

if __name__ == "__main__":
    queries, ground_truths = load_evaluation_data()

    print("\n🔍 MODEL EVALUATION")

    cosine_similarity_score = evaluate_similarity_cosine(queries, ground_truths)

    print("\n📊 OVERALL RESULTS:")
    print(f"✅ Average Adjusted Cosine Similarity Score: {cosine_similarity_score:.2f}")
