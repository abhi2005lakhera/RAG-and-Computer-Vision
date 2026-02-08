import os
import pickle
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List
import torch
torch.set_num_threads(1)


DATA_DIR = "data/documents"
VECTOR_DIR = "vectorstore/faiss_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def load_documents():
    texts = []

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)

        if file.endswith(".pdf"):
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():   # IMPORTANT
                    texts.append(text)

        elif file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                if text and text.strip():   # IMPORTANT
                    texts.append(text)

    return texts



def chunk_text(text):
    if not text or not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks



def ingest_documents():

    print("Loading documents...")
    raw_docs = load_documents()

    print("Chunking documents...")
    chunks = []
    for doc in raw_docs:
        chunks.extend(chunk_text(doc))

    print(f"Total chunks created: {len(chunks)}")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Generating embeddings...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    if len(chunks) == 0:
        raise ValueError("No valid text chunks found. Check your documents.")


    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))

    with open(os.path.join(VECTOR_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    ingest_documents()

