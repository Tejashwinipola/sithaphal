from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3

# Step 1: Extract text from PDF
pdf_reader = PdfReader("sample.pdf")
text = " ".join([page.extract_text() for page in pdf_reader.pages])

# Step 2: Chunk text
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# Step 3: Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Step 4: Save embeddings in a database
conn = sqlite3.connect("embeddings.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, text TEXT, vector BLOB)")
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    cursor.execute("INSERT INTO embeddings (id, text, vector) VALUES (?, ?, ?)", (i, chunk, embedding.tobytes()))
conn.commit()
conn.close()
