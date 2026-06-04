import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil

load_dotenv()

# A1: Load documentation
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Sanity Check
    print(f"Total character count: {len(text)}")
    print(f"Sample text (first 500 chars):\n{text[:500]}")
    return text

# A2: Chunking with overlap
def chunk_document(text, chunk_size=1000, chunk_overlap=150):
    """
    Why overlap is important for technical code snippets:
    - Prevents context fragmentation (e.g., function definitions split from their body)
    - Ensures code blocks remain interpretable across chunk boundaries
    - Preserves API endpoint paths that might be cut mid-string
    Chunk size 1000 ensures full sections (e.g., all 4 grant types) stay together.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    return chunks

# A3: Vector storage — embed chunks and store in ChromaDB
def create_vector_store(chunks, persist_directory="./chroma_db"):
    # Delete old DB so chunks are fresh
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("Deleted old ChromaDB.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    print("Vector store created and persisted.")
    return vector_store

if __name__ == "__main__":
    text = load_document("upwork_api_reference.txt")
    chunks = chunk_document(text)
    vector_store = create_vector_store(chunks)