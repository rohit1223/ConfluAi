"""
This module demonstrates how to load, clean, and process text documents from a specified directory,
and compute their embeddings using the Hugging Face model "BAAI/bge-small-en-v1.5". The script loads
the documents using utility functions, cleans them using a custom cleaning function, tokenizes the text,
and computes normalized embeddings which are then printed to the console.
"""
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

from utils.utils import load_documents, clean_confluence_text
from config import DOCS_DIR

MODEL_NAME = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
load_documents(DOCS_DIR)


raw_docs = load_documents(DOCS_DIR)
# documents = []
documents = [clean_confluence_text(doc) for doc in raw_docs]

# Currently using SimpleDirectoryReader to load documents
# documents = SimpleDirectoryReader(DOCS_DIR).load_data()
# for doc in raw_docs:
# cleaned_doc = clean_confluence_text(raw_docs)
# chunks = chunk_document(cleaned_doc, 500, 50)
# for chunk in chunks:
#     documents.append(Document(text=chunk))


# text = "This is a test sentence."
inputs = tokenizer(
    documents,
    padding=True,
    truncation=True,
    return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
embedding_np = embedding.cpu().numpy()
normalized_vec = normalize(embedding_np)[0]
print("Embedding:", normalized_vec)
