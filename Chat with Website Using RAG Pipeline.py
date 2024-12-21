import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Query for similar chunks
def find_similar(query, top_k=3):
    conn = sqlite3.connect("embeddings.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, vector FROM embeddings")
    data = cursor.fetchall()
    
    query_embedding = model.encode([query])[0]
    similarities = []
    for id_, text, vector in data:
        embedding = np.frombuffer(vector, dtype=np.float32)
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities.append((id_, text, similarity))
    
    conn.close()
    return sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]

# Example Query
results = find_similar("What is the unemployment rate based on degree?")
for result in results:
    print(f"ID: {result[0]}, Text: {result[1]}, Similarity: {result[2]}")
