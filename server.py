from flask import Flask, request, jsonify
import pickle
import openai
import numpy as np

app = Flask(__name__)

# Load embeddings
with open("../embeddings/law_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

openai.api_key = "YOUR_OPENAI_API_KEY"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_most_relevant(query, top_n=3):
    # Get embedding of query
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response['data'][0]['embedding']
    
    # Compute similarity
    similarities = [cosine_similarity(query_embedding, e['embedding']) for e in embeddings]
    
    # Get top N chunks
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [embeddings[i]['text'] for i in top_indices]

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    user_question = data.get("question", "")
    relevant_chunks = get_most_relevant(user_question)
    
    # Generate GPT answer
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question using the legal context below:\n\n{context}\n\nQuestion: {user_question}\nAnswer:"
    
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = completion.choices[0].message['content']
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
