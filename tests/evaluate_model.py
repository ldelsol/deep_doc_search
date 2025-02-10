import numpy as np
from rouge_score import rouge_scorer
from langchain_huggingface import HuggingFaceEmbeddings
from deep_doc_search.query_handler import search_in_vector_store
from deep_doc_search.llm_handler import generate_response

# Load the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-distilroberta-v1")

def normalize_text(text):
    """Cleans text by removing extra spaces, line breaks, and converting to lowercase."""
    return " ".join(text.lower().strip().split())

def evaluate_recall_k(queries, ground_truths, k=3):
    """Evaluates the model's precision using Recall@k and logs query retrieval details."""
    correct = 0
    total = len(queries)
    details = []

    for query, expected_answer in zip(queries, ground_truths):
        results, _ = search_in_vector_store(query, k)

        # Normalize text for comparison
        expected_answer_norm = normalize_text(expected_answer)

        # Check if the correct response is among the top k results
        found = False
        rank = -1  

        for i, res in enumerate(results):
            if expected_answer_norm in normalize_text(res):
                found = True
                rank = i + 1  # 1-based index
                break  

        # Store results
        if found:
            correct += 1
            details.append(f"✅ Query: '{query}' → Found at position {rank} / {k}")
        else:
            details.append(f"❌ Query: '{query}' → Not found in top {k}")

    recall_k = correct / total

    # Display detailed results
    print("\n📊 Recall@k Details:")
    for detail in details:
        print(detail)

    print(f"\n📊 Recall@{k}: {recall_k:.2f}")
    return recall_k

def evaluate_rouge_l(queries, ground_truths):
    """Evaluates the LLM-generated responses using ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for query, expected_answer in zip(queries, ground_truths):
        generated_response = generate_response(query)  

        # Compute ROUGE-L score
        rouge_l = scorer.score(expected_answer, generated_response)['rougeL'].fmeasure
        scores.append(rouge_l)

        # If ROUGE-L is low, use LLM to validate the response
        if rouge_l < 0.6:
            is_valid = llm_validate_answer(expected_answer, generated_response)
            if is_valid:
                print(f"⚠️ Low ROUGE-L ({rouge_l:.2f}), but the LLM validated the response as correct.")
            else:
                print(f"❌ Response deemed incorrect by the LLM (ROUGE-L = {rouge_l:.2f}).")

    avg_rouge_l = np.mean(scores)
    print(f"📊 Average ROUGE-L Score: {avg_rouge_l:.2f}")
    return avg_rouge_l

def llm_validate_answer(reference, generated):
    """Uses an LLM to verify whether the generated response contains the correct information."""
    validation_prompt = f"""
    You are an evaluator for AI-generated answers. Your task is to check if the generated response contains the key information from the reference answer.
    
    **Reference answer:**
    {reference}

    **Generated response:**
    {generated}

    Does the generated response contain the correct information? Answer with 'Yes' or 'No'.
    """
    evaluation_result = generate_response(validation_prompt)  
    return "yes" in evaluation_result.lower()

if __name__ == "__main__":
    # List of test queries and ground truth answers
    queries = [
        "What is the plan to protect water resources?",
        "Who is the CEO of LVMH?",
        "Which countries generate the most revenue for LVMH?"
    ]
    ground_truths = [
        "The goal is a 30% reduction in the amount of water used by LVMH’s operations and its value chain by 2030",
        "Bernard Arnault",
        "United States"
    ]

    print("\n🔍 MODEL EVALUATION")
    
    # Recall@k evaluation
    recall_score = evaluate_recall_k(queries, ground_truths, k=3)

    # ROUGE-L evaluation with LLM validation
    rouge_l_score = evaluate_rouge_l(queries, ground_truths)

    # Summary of results
    print("\n📊 OVERALL RESULTS:")
    print(f"✅ Recall@3: {recall_score:.2f}")
    print(f"✅ Average ROUGE-L Score: {rouge_l_score:.2f}")
