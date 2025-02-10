import ollama

def generate_response(prompt):
    """Generates a response using Mistral locally via Ollama."""
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    query = "Explain FAISS in one sentence."
    response = generate_response(query)
    print("\nMistral Response:")
    print(response)
