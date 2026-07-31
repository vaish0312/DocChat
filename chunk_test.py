from pypdf import PdfReader

# --- 1) read all text out of the PDF ---
reader = PdfReader("sample.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

print("Total characters in document:", len(text))

# --- 2) chop the text into overlapping chunks ---
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap   # step back a little so chunks overlap
    return chunks

chunks = chunk_text(text)
print("Number of chunks:", len(chunks))
print("\n--- First chunk ---")
print(chunks[0])
