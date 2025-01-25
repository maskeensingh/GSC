import fitz  # PyMuPDF
import logging
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import tiktoken

def extract_text_from_pdfs(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text") or ""
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
    return text

def clean_pdf_text(text):
    """Remove irrelevant sections like acknowledgments, prefaces, and exam questions."""
    filtered_text = [
        line.strip() for line in text.split("\n")
        if not any(keyword.lower() in line.lower() for keyword in ["acknowledgement", "preface", "table", "exam"])
    ]
    return " ".join(filtered_text)

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

def generate_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = split_text_into_chunks(text)
    if not chunks:
        logging.error("No chunks generated for embedding.")
        return [], None
    embeddings = model.encode(chunks)
    return chunks, embeddings

def create_faiss_index(embeddings):
    if embeddings.size == 0:
        logging.error("No embeddings provided to create FAISS index.")
        return None

    dim = embeddings.shape[1]  # Dimension of the embeddings
    nlist = min(100, embeddings.shape[0])  # Ensure nlist <= number of training points

    # Create the FAISS index
    quantizer = faiss.IndexFlatL2(dim)  # Quantizer for IVF
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # Use nlist instead of a fixed value

    # Train the index
    if embeddings.shape[0] >= nlist:  # Ensure there are enough training points
        index.train(embeddings)
    else:
        logging.warning(f"Not enough training points ({embeddings.shape[0]}) for {nlist} clusters. Using a flat index instead.")
        index = faiss.IndexFlatL2(dim)  # Fallback to a flat index

    # Add embeddings to the index
    index.add(embeddings)
    return index

def get_top_chunks(query, index, pdf_chunks, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=top_k)
    top_chunks = [pdf_chunks[i] for i in I[0] if i < len(pdf_chunks)]
    return top_chunks

def count_tokens(text):
    """Count the number of tokens in the input text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def safe_summarize(content, token_limit=2000):
    """Summarize content to stay within token limits."""
    if count_tokens(content) > token_limit:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer(content, max_length=300, min_length=50, do_sample=False)[0]['summary_text']
    return content
