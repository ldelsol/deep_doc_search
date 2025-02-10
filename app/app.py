import streamlit as st
from deep_doc_search.query_handler import search_in_vector_store
from deep_doc_search.llm_handler import generate_response

st.title("📚 Chatbot FAQ on Internal Document")

user_query = st.text_input("Ask your question about the document:", "")

if user_query:
    # Retrieve only the textual results
    search_results, _ = search_in_vector_store(user_query, k=3)  # Ignoring distances

    if search_results:
        context = "\n\n".join(search_results)  

        prompt = f"""
        You are an AI assistant specialized in analyzing internal documents.
        Here is an excerpt from the document that may help answer the question:

        {context}

        Question: {user_query}
        Respond accurately and concisely using only the provided information.
        """

        response = generate_response(prompt)
        st.markdown(f"### 🤖 Response:\n{response}")

    else:
        st.warning("⚠️ No relevant information found in the document.")
