import streamlit as st
import requests
import urllib.request
import time
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Initialize vector store
@st.cache_resource
def init_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

# B1: Semantic retrieval — fetches top-k relevant chunks from ChromaDB
def retrieve_context(query, vector_store, k=5):
    results = vector_store.similarity_search(query, k=k)
    chunks = [doc.page_content for doc in results]
    return chunks

# B2: LLM API call — DeepInfra (OpenAI-compatible endpoint)
def ask_llm(query, context_chunks, system_prompt):
    context = "\n\n---\n\n".join(context_chunks)

    user_prompt = f"""Context from Upwork API docs:
{context}

Question: {query}

Instructions:
- Answer ONLY using the context above.
- If the answer is not present, say EXACTLY: "I'm sorry, but the provided documentation does not contain that information."
- Be concise and technical."""

    headers = {
        "Authorization": f"Bearer {os.getenv('LLM_API_KEY')}",
        "Content-Type": "application/json"
    }

    # DeepInfra uses OpenAI-compatible chat completions format
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.1
    }

    # Auto-detect system proxy
    proxies = urllib.request.getproxies()

    start_time = time.time()
    try:
        response = requests.post(
            os.getenv('LLM_API_ENDPOINT'),
            headers=headers,
            json=payload,
            proxies=proxies,
            timeout=60
        )
        latency = time.time() - start_time

        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
        elif response.status_code == 401:
            answer = "❌ Invalid API Key. Check your LLM_API_KEY in .env"
            latency = None
        elif response.status_code == 503:
            answer = "⏳ Model is loading. Please wait and try again."
            latency = None
        else:
            answer = f"❌ API Error {response.status_code}: {response.text}"
            latency = None

    except requests.exceptions.ConnectionError as e:
        latency = None
        answer = f"❌ Cannot reach the API.\n\nDetail: {str(e)}"
    except requests.exceptions.Timeout:
        latency = None
        answer = "❌ Request timed out. Try again."

    return answer, latency, context_chunks


# Streamlit UI
def main():
    st.set_page_config(page_title="Upwork API Support Bot", layout="wide")
    st.title("🤖 Upwork API Technical Support Bot")
    st.markdown("Ask any question about the Upwork API based on the official documentation.")

    SYSTEM_PROMPT = """You are a Senior Upwork API Consultant.
You are strict, precise, and only answer based on official documentation.
Never hallucinate. Never use external knowledge about Upwork not present in the provided context."""

    vector_store = init_vector_store()

    query = st.text_input(
        "Ask a question about the Upwork API:",
        placeholder="e.g., How long is an OAuth access token valid for?"
    )

    if query:
        with st.spinner("🔍 Retrieving relevant documentation..."):
            context_chunks = retrieve_context(query, vector_store, k=5)

        with st.spinner("🤔 Generating answer..."):
            answer, latency, used_chunks = ask_llm(query, context_chunks, SYSTEM_PROMPT)

        st.markdown("### 💬 Answer")
        st.info(answer)

        if latency:
            st.metric("⏱️ Response Latency", f"{latency:.2f} seconds")

        with st.expander("📚 Sources (exact snippets used)"):
            for i, chunk in enumerate(used_chunks, 1):
                st.markdown(f"**Source {i}:**")
                st.code(chunk, language="text")
                st.divider()


if __name__ == "__main__":
    main()