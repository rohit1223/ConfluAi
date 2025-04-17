"""
This module provides utility functions for loading and processing text documents.
It includes functions to load documents from a directory with proper encoding detection,
and to split long text documents into smaller chunks using a recursive character splitter.
"""
import os
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


def chunk_document(text: str, chunk_size: int = 1000,
                   chunk_overlap: int = 100) -> list:
    """
    Splits the input text into chunks using a recursive character text splitter.

    Args:
        text (str): The raw text to split.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks
