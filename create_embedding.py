import os
from openai import OpenAI
import tiktoken
import pickle

# Config
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
TEXT_FILE = "../embeddings/all_laws.txt"
EMBEDDINGS_FILE = "../embeddings/law_embeddings.pkl"
CHUNK_SIZE = 500  # words per chunk

# Load text
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks
words = text.split()
chunks = [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

# Generate embeddings
embeddings = []
for chunk in chunks:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )
    embeddings.append({
        "text": chunk,
        "embedding": response.data[0].embedding
    })

# Save embeddings
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print(f"{len(embeddings)} chunks embedded and saved!")
