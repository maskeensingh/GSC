import streamlit as st
import os
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import faiss
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import tiktoken
from transformers import pipeline
from langchain.chains import LLMChain

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")  # Updated model name

# Token Counting Function
def count_tokens(text):
    """Count the number of tokens in the input text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# Safe Summarization Function
def safe_summarize(content, token_limit=2000):
    """Summarize content to stay within token limits."""
    if count_tokens(content) > token_limit:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer(content, max_length=300, min_length=50, do_sample=False)[0]['summary_text']
    return content

# Read and Merge CSV Data
def read_and_merge_csv():
    """Read and merge Training.csv and Testing.csv into a single DataFrame."""
    try:
        # Load the CSV files
        training_df = pd.read_csv("Training.csv")
        testing_df = pd.read_csv("Testing.csv")
        
        # Merge the two DataFrames
        merged_df = pd.concat([training_df, testing_df], ignore_index=True)
        return merged_df
    except Exception as e:
        logging.error(f"Error reading or merging CSV files: {e}")
        return None

# Generate Embeddings using SentenceTransformer
def generate_embeddings(texts):
    """Generate embeddings for the text using SentenceTransformer."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

# Create FAISS Index
def create_faiss_index(embeddings):
    if embeddings.size == 0:
        logging.error("No embeddings provided to create FAISS index.")
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Search for Relevant Content using FAISS
def get_top_chunks(query, index, questions, top_k=5):
    """Search for the most relevant content in the CSV using FAISS."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=top_k)
    top_chunks = [questions[i] for i in I[0]] if I[0].size > 0 else []
    return top_chunks

# Build a prompt template for answering questions on menstruation
menstruation_prompt = """
You are a knowledgeable assistant specializing in menstrual health. You will be given a context containing relevant information from reliable sources about menstruation, and your task is to provide accurate, helpful, and concise answers to the questions.
Note that user asking questions may be of any age so answer in a way they can understand better. You may use examples and make it better informative.
Context: {context}

Question: {question}

Answer:
"""

# Set up the LLMChain for the menstrual health Q&A
def build_menstruation_qa_chain():
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=menstruation_prompt
    )
    # Wrap the prompt with the LLM to create a chain
    return LLMChain(prompt=prompt, llm=llm)

# Initialize the Q&A Chain
qa_chain = build_menstruation_qa_chain()

# Streamlit app
st.title("Cycle Wise")
st.markdown("This app helps answer questions about menstrual health using a retrieval-augmented pipeline.")

# Read and merge CSV files
merged_df = read_and_merge_csv()

if merged_df is not None:
    # Ensure the CSV contains 'question' and 'answer' columns
    if "question" not in merged_df.columns or "answer" not in merged_df.columns:
        st.error("CSV files must contain 'question' and 'answer' columns.")
    else:
        # Generate Embeddings for questions
        questions = merged_df['question'].tolist()
        embeddings = generate_embeddings(questions)

        # Create FAISS index for retrieval
        faiss_index = create_faiss_index(embeddings)

        # User input for query
        query = st.text_input("Enter your question about menstrual health:")
        if query:
            # Retrieve relevant chunks
            top_chunks = get_top_chunks(query, faiss_index, questions)

            # Combine the top chunks into a context
            context = "\n".join(top_chunks)

            # Answer the query using the menstruation Q&A chain
            response = qa_chain.run({"context": context, "question": query})

            # Display the result
            st.markdown("### Answer:")
            st.write(response)
else:
    st.error("Failed to load or merge CSV files. Please ensure 'Training.csv' and 'Testing.csv' are present.")