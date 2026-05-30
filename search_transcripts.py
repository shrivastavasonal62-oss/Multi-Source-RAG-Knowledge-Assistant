import joblib

art = joblib.load("vectors.joblib")
texts = art["texts"]

keyword = "recursion"

found = False

for i, t in enumerate(texts):
    if keyword.lower() in t.lower():
        print(f"\n----- Chunk {i} -----")
        print(t[:1000])
        found = True

if not found:
    print("No recursion content found.")