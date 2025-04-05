# utils.py
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chardet

def load_documents(doc_dir: str) -> list:
    """
    Load all .txt and .md files from the specified directory,
    automatically detecting file encoding.
    """
    docs = []
    for file in os.listdir(doc_dir):
        if file.endswith(".txt") or file.endswith(".md"):
            filepath = os.path.join(doc_dir, file)
            with open(filepath, "rb") as f:
                raw_data = f.read()
            encoding_info = chardet.detect(raw_data)
            encoding = encoding_info.get("encoding", "utf-8")
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    docs.append(f.read())
            except UnicodeDecodeError:
                # Fallback to latin-1 if needed
                with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
                    docs.append(f.read())
    return docs

# cleaning logic
def clean_confluence_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
    text = re.sub(r"\{[^}]+\}", "", text)  # Remove wiki macros or curly braces
    text = re.sub(r"(\*{2,}|\_{2,}|-{2,})", "", text)  # Remove Markdown-style formatting
    text = re.sub(r"(?i)page created by.*|last edited by.*", "", text)
    text = re.sub(r"\s{2,}", " ", text)  # Normalize extra whitespace
    return text.strip()


def chunk_document(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
    """
    Splits the input text into chunks using a recursive character text splitter.

    Args:
        text (str): The raw text to split.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks
