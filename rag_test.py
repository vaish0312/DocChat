import chromadb
from pypdf import PdfReader

# --- 1) read the PDF ---
reader = PdfReader("sample.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# --- 2) chunk it (same logic as before) ---
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(text)
print("Chunks created:", len(chunks))

# --- 3) store chunks in the vector database (auto-embeds them) ---
client = chromadb.Client()
collection = client.get_or_create_collection("docchat")
collection.add(
    documents=chunks,
    ids=[f"chunk-{i}" for i in range(len(chunks))]
)
print("Stored", collection.count(), "chunks in the vector database.")

# --- 4) retrieve: find the chunks most relevant to a question ---
question = "What is the transformer architecture?"

results = collection.query(
    query_texts=[question],
    n_results=3
)

print("\nQUESTION:", question)
print("\n--- Top 3 most relevant chunks ---")
for i, chunk in enumerate(results["documents"][0]):
    print(f"\n[Chunk {i+1}]")
    print(chunk[:300])   # first 300 chars of each


# --- 5) generate: give the retrieved chunks to the LLM to write an answer ---
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

# combine the retrieved chunks into one block of context
context = "\n\n".join(results["documents"][0])

prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

response = client_ai.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
)

print("\n=== ANSWER ===")
print(response.choices[0].message.content)