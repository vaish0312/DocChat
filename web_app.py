# fix for ChromaDB on Streamlit Cloud (swaps in a newer sqlite) - harmless locally
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import chromadb
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq

# --- page setup (must be the first Streamlit command) ---
st.set_page_config(page_title="DocChat — Chat with your documents", page_icon="📄", layout="centered")

# --- a little styling ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stChatMessage"] { border-radius: 14px; padding: 6px 10px; }
    .stChatInput textarea { border-radius: 12px; }
    .hero-title { text-align:center; font-size:2.5rem; font-weight:800;
        background: linear-gradient(90deg,#6366f1,#8b5cf6);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0; }
    .hero-sub { text-align:center; color:#8b93a7; margin-top:4px; font-size:1rem; }
</style>
""", unsafe_allow_html=True)

# --- setup ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = st.secrets["GROQ_API_KEY"]
ai = Groq(api_key=api_key)

def chunk_text(text, size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# --- header ---
st.markdown('<div class="hero-title">📄 DocChat</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a document and chat with it — answers grounded in your file, with sources.</div>', unsafe_allow_html=True)
st.write("")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- sidebar: upload + info ---
with st.sidebar:
    st.header("📄 DocChat")
    st.caption("Chat with any PDF using AI.")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Upload a PDF\n2. Ask a question\n3. Get a grounded answer with sources")
    st.markdown("---")
    st.caption("Built with Python · Streamlit · ChromaDB · Llama 3.3 (Groq)")

# --- index the uploaded file once ---
if uploaded and st.session_state.get("loaded_file") != uploaded.name:
    with st.spinner("Reading and indexing your document…"):
        reader = PdfReader(uploaded)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        chunks = chunk_text(text)
        client = chromadb.Client()
        try:
            client.delete_collection("docchat")
        except Exception:
            pass
        collection = client.get_or_create_collection("docchat")
        collection.add(documents=chunks, ids=[f"c{i}" for i in range(len(chunks))])
        st.session_state.chroma_client = client
        st.session_state.collection = collection
        st.session_state.loaded_file = uploaded.name
        st.session_state.messages = []
    st.success(f"'{uploaded.name}' loaded — {len(chunks)} chunks. Ask away!")

# --- show chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- chat input ---
if "collection" in st.session_state:
    question = st.chat_input("Ask a question about your document…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        results = st.session_state.collection.query(query_texts=[question], n_results=4)
        context = "\n\n".join(results["documents"][0])
        prompt = f"""You are a helpful assistant answering questions about a document.
Use ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = ai.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                ).choices[0].message.content
                st.markdown(answer)
                with st.expander("📄 Sources"):
                    for i, c in enumerate(results["documents"][0]):
                        st.markdown(f"**Passage {i+1}:** {c[:300]}…")
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload a PDF in the sidebar to start chatting.")