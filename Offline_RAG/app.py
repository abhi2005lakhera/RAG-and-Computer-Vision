import os
import streamlit as st
from rag_pipeline import generate_answer
from ingest import ingest_documents


UPLOAD_DIR = "data/documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Offline RAG Chatbot", layout="wide")

st.title("📚 Fully Offline RAG Chatbot")
st.caption("Uses local embeddings, FAISS, and a local LLM (no APIs)")

st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        save_path = os.path.join(UPLOAD_DIR, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("Files uploaded successfully!")

if st.button("Rebuild Knowledge Base"):
    with st.spinner("Indexing documents..."):
        ingest_documents()
    st.cache_resource.clear()
    st.success("Documents indexed successfully!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question based on your documents:")

if query:
    answer, sources = generate_answer(query)

    st.session_state.chat_history.append((query, answer))

    st.markdown("### 🤖 Answer")
    st.write(answer)

    with st.expander("🔍 Retrieved Context"):
        for i, chunk in enumerate(sources, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(chunk)

st.markdown("---")

st.markdown("### 💬 Chat History")
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")



