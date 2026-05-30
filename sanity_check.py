import joblib, numpy as np, faiss
from sentence_transformers import SentenceTransformer

VEC_PATH = "vectors.joblib"
art = joblib.load(VEC_PATH)   # expects keys: "embeddings", "texts"
embs  = art["embeddings"].astype("float32")
texts = art["texts"]

faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1]); index.add(embs)

encoder = SentenceTransformer("all-MiniLM-L6-v2")
query = "Explain Python functions with an example."
qv = encoder.encode([query], convert_to_numpy=True).astype("float32")
faiss.normalize_L2(qv)
D, I = index.search(qv, 5)

print("\nTop matches:\n")
for r, i in enumerate(I[0], 1):
    snippet = texts[i][:220].replace("\n"," ")
    print(f"{r}. score={D[0][r-1]:.3f}  {snippet}…")
