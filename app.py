import faiss
import numpy as np
import torch
import openai
import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import re


# Load the FAISS index and precomputed embeddings
index = faiss.read_index("faiss_index.index")
corpus_embeddings = np.load("corpus_embeddings.npy")
with open("scraped_data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
entries = raw_text.split("\n\n")

# Function to clean text content
def clean_text_content(text):
    patterns_to_remove = [
        r"Up\^Add To My Favorites",
        r"Add To My Favorites",
        r"Up",
        r"\[.*?\]",
        r"Code Text.*?:",
        r"DIVISION\s*\d+.*?CHAPTER\s*\d+",
        r"https?://\S+",
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to extract entry details
def extract_entry_details(entry):
    url_pattern = r"URL:\s*(https?://\S+)"
    code_pattern = r"DIVISION\s*(\d+).*(CHAPTER\s*\d+)"
    text_pattern = r"Text Content:\s*(.*)"

    url = re.search(url_pattern, entry)
    code = re.search(code_pattern, entry)
    text = re.search(text_pattern, entry)

    url = url.group(1) if url else None
    code = code.group(0) if code else None
    text = clean_text_content(text.group(1)) if text else None

    return {"url": url, "code": code, "text": text}

# Extract all entries
documents = [extract_entry_details(entry) for entry in entries]

# Remove entries with missing text content
documents = [doc for doc in documents if doc["text"]]

# Create a Hugging Face Dataset with structured data
from datasets import Dataset
corpus = Dataset.from_dict({
    "url": [doc["url"] for doc in documents],
    "code": [doc["code"] for doc in documents],
    "text": [doc["text"] for doc in documents]
})

# Load embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

# Embedding function with GPU support
def encode(texts, batch_size=8):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

# Set up OpenAI GPT-4 API key
openai.api_key =

def summarize_text(text, max_summary_tokens=1024):
    # Prepare the prompt with the text to summarize
    prompt = f"Summarize the following text:\n\n{text}"

    # Use openai.ChatCompletion.create() for summarization
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-4-32k" for handling larger inputs
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_summary_tokens,  # Limit the number of tokens in the summary
        temperature=0.5,  # Control the randomness (set to 0 for deterministic output)
        top_p=1.0,  # Top probability sampling
        n=1,  # Single response  # Use the specified stop token
    )

    # Extract the summary from the response
    summary = response['choices'][0]['message']['content'].strip()
    return summary

# Updated retrieval function to return full document details (url, code, division, text)
def retrieve(query, top_k=3):
    query_embedding = encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    # Fetch the corresponding document details
    results = []
    for i in indices[0]:
        doc = {
            "url": corpus["url"][i],
            "code": corpus["code"][i],
            "text": corpus["text"][i]
        }
        results.append(doc)
    
    return results

# Load generative model
gen_model_name = "google/flan-t5-large"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to("cuda")

# Adjust the response generation with more detailed context and parameters like temperature
def generate_response(query, documents, max_length=500, temperature=0.7, top_p=0.9):
    context = " ".join([doc["text"] for doc in documents]) # Extract "text" from each document
    input_text = f"Query: {query}\nContext: {context}\nAnswer with details:"
    inputs = gen_tokenizer(input_text, return_tensors="pt", truncation=True).to("cuda")  # Move to GPU
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs, 
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Updated RAG pipeline with more explicit instruction
def rag_pipeline(query):
    print("Starting RAG pipeline...")
    # Retrieve relevant documents
    documents = retrieve(query)
    print(f"Retrieved {len(documents)} documents.")
    
    # Generate a response with improved context handling
    #response = generate_response(query, documents, max_length=300, temperature=0.7, top_p=0.9)
    #print("Generated response.")

    # Generate a summary for each document
    summaries = []
    for doc in documents:
        summary = summarize_text(doc["text"])
        summaries.append({
            "url": doc["url"],
            "code": doc["code"],
            "summary": summary
        })  
	
	 
    print("Generated summaries.")
    return summaries

# Example usage
query = "When can I enforce an eviction?"
retrieved_docs = retrieve(query)

# Display the retrieved documents and generate summaries
for doc in retrieved_docs:
    # Output the actual text, code, and division (if present)
    print(f"URL: {doc['url']}")
    print(f"Code: {doc['code']}")
    
    # If division is available, print it
    if "division" in doc:
        print(f"Division: {doc['division']}")
    
    # Display the actual text content
    print(f"Text: {doc['text']}")
    
    # Generate a summary using GPT-4 model
    summary = summarize_text(doc["text"])
    
    # Output the summary
    print(f"Summary: {summary}")
    print("-" * 80)  # Separator for clarity

# Streamlit interface
st.title("Legal Assistance Chatbot")

user_input = st.text_input("Ask me anything:")

if user_input:
    st.write("Query:", user_input)
    with st.spinner("Processing your query..."):
        # Get the summarized results from the RAG pipeline
        response = rag_pipeline(user_input)
        
        # Display each document's URL, code, and summary
        for doc in response:
            st.write(f"### URL: {doc['url']}")
            st.write(f"**Code**: {doc['code']}")
            st.write(f"**Summary**: {doc['summary']}")
            st.write("-" * 80)  # Separator between results