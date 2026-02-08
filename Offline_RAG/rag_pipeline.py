import faiss
import pickle
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from pathlib import Path

VECTOR_DIR = "vectorstore/faiss_index"
MODEL_PATH = "models/orca-mini-3b-gguf2-q4_0.gguf"
TOP_K = 4

MODEL_DIR = Path(__file__).parent / "models"

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index and chunks
def load_index():
    return faiss.read_index(f"{VECTOR_DIR}/index.faiss")

def load_chunks():
    with open(f"{VECTOR_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return chunks

# Load local GPT4All model
llm = GPT4All(
    model_name="orca-mini-3b-gguf2-q4_0.gguf",
    model_path=str(MODEL_DIR),
    allow_download=False,
    device = "cpu"
)

def retrieve(query: str, index, chunks):
    query_embedding = embedder.encode([query])
    _, indices = index.search(query_embedding, TOP_K)
    return [chunks[i] for i in indices[0]]

def generate_answer(query: str):
    index = load_index()
    chunks = load_chunks()
    context_chunks = retrieve(query, index, chunks)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an assistant that answers ONLY using the given context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    with llm.chat_session():
        response = llm.generate(prompt, max_tokens=300)

    return response.strip(), context_chunks
