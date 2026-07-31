# DocChat 📄 — Chat with your documents (RAG)

Upload a PDF and ask questions in plain English. DocChat finds the relevant passages and answers
using **only** your document — with sources shown, so you can trust the answer.

## What it does
- Upload any PDF, then chat with it like ChatGPT
- Uses **Retrieval-Augmented Generation (RAG)**: embeds your document into a vector database,
  retrieves the most relevant passages for each question, and has an LLM answer from them
- Shows the source passages behind every answer (grounded — reduces hallucination)

## How it works
1. **Chunk** the document into overlapping pieces
2. **Embed** each chunk into a vector database (ChromaDB)
3. **Retrieve** the passages most similar to your question (semantic search)
4. **Generate** an answer with an LLM (Llama 3.3 via Groq), using only those passages

## Tech stack
Python · Streamlit · ChromaDB (vector DB + embeddings) · Llama 3.3 (Groq) · pypdf

## Run it locally
\`\`\`bash
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env   # free key at console.groq.com
streamlit run web_app.py
\`\`\`

## What I learned
- Embeddings & semantic search — matching by meaning, not keywords
- Chunking strategy (size + overlap) and why it matters
- Vector databases and retrieval
- Grounding answers in retrieved context to reduce hallucination

*Built as a hands-on RAG project while studying LLMs and applied AI. — Vaishnavi Patil*