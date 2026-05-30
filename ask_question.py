import joblib
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

OLLAMA_MODEL = "llama3.2:1b"
TOP_K = 5

print("\n📘 Welcome to your RAG-Based AI Teaching Assistant!")
print("This assistant answers questions using Python course transcript.")
print("------------------------------------------------------------")

# ---------------- LOAD VECTOR FILE ----------------
try:
    art = joblib.load("vectors.joblib")

    if "embeddings" not in art or "texts" not in art:
        raise KeyError("vectors.joblib must contain 'embeddings' and 'texts' keys")

    embs = np.array(art["embeddings"]).astype("float32")
    texts = art["texts"]

    print(f"✅ Loaded {len(texts)} transcript chunks.")
    print(f"✅ Embedding shape: {embs.shape}")

except Exception as e:
    print(f"❌ Error loading vectors.joblib: {e}")
    exit()


# ---------------- BUILD FAISS INDEX ----------------
faiss.normalize_L2(embs)

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

print("✅ FAISS index created successfully.")


# ---------------- LOAD EMBEDDING MODEL ----------------
print("⏳ Loading embedding model...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded successfully.")


# ---------------- SAMPLE QUESTIONS ----------------
sample_questions = [
    "Explain Python functions with an example",
    "What are loops in Python?",
    "How do conditional statements work?",
    "What is list comprehension in Python?",
    "Explain the use of dictionaries in Python",
    "What is recursion?",
    "How do you handle exceptions in Python?",
    "Explain object-oriented programming concepts",
    "What is the difference between tuples and lists?",
    "Explain how for and while loops differ"
]

print("\n💡 Try asking one of these sample questions:\n")
for q in sample_questions:
    print(f"  ➤ {q}")

print("\n(Type 'exit' or 'quit' anytime to stop.)")


# ---------------- CHECK OLLAMA ----------------
def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available = [m["name"] for m in models]

            print("\n✅ Ollama is running.")
            print("📦 Available Ollama models:")
            for model in available:
                print(f"  - {model}")

            if OLLAMA_MODEL not in available:
                print(f"\n⚠️ Warning: '{OLLAMA_MODEL}' not found in Ollama.")
                print(f"Run this command first:")
                print(f"ollama pull {OLLAMA_MODEL}")

            return True

    except requests.exceptions.ConnectionError:
        print("\n❌ Ollama is not running.")
        print("Start Ollama first, then run this file again.")
        print("Or run this command in terminal:")
        print("ollama serve")
        return False

    except Exception as e:
        print(f"\n❌ Error checking Ollama: {e}")
        return False


check_ollama()


# ---------------- RETRIEVAL FUNCTION ----------------
def retrieve_context(query, top_k=TOP_K):
    qv = encoder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qv)

    scores, ids = index.search(qv, top_k)

    retrieved_chunks = []
    source_info = []

    for rank, idx in enumerate(ids[0], 1):
        if idx == -1:
            continue

        chunk = texts[idx]
        score = float(scores[0][rank - 1])

        retrieved_chunks.append(f"[Chunk {rank}]\n{chunk}")
        source_info.append((rank, score, chunk))

    context = "\n\n".join(retrieved_chunks)
    return context, source_info


# ---------------- OLLAMA ANSWER FUNCTION ----------------
def ask_ollama(question, context):
    prompt = f"""
You are a helpful AI Python teaching assistant.

Rules:
1. Use only the transcript context given below.
2. Explain in simple beginner-friendly language.
3. If the context does not contain enough information, say:
   "The transcript does not contain enough information about this."
4. Add a short Python example only if it is relevant.
5. Keep the answer clear and not too long.

Transcript Context:
{context}

Student Question:
{question}

Final Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 300
                }
            },
            timeout=180
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"❌ Ollama error: {response.text}"

    except requests.exceptions.ConnectionError:
        return "❌ Ollama is not running. Open Ollama first, then run this file again."

    except requests.exceptions.Timeout:
        return "❌ Ollama took too long to respond. Try using a smaller model or shorter context."

    except Exception as e:
        return f"❌ Error generating answer: {e}"


# ---------------- MAIN LOOP ----------------
while True:
    query = input("\n🔹 Ask your question: ").strip()

    if query.lower() in ["exit", "quit"]:
        print("\n👋 Exiting assistant. Goodbye!")
        break

    if not query:
        print("⚠️ Please type a question.")
        continue

    print("\n🔍 Retrieving relevant transcript chunks...")
    context, sources = retrieve_context(query)

    print("🤖 Generating answer...\n")
    answer = ask_ollama(query, context)

    print("✅ Answer:\n")
    print(answer)

    print("\n📌 Sources used:")
    for rank, score, chunk in sources:
        snippet = chunk[:180].replace("\n", " ")
        print(f"{rank}. score={score:.3f} | {snippet}...")

    print("\n---------------------------------------------")