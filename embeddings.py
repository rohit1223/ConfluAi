# embeddings.py
import torch
from llama_index.core.base.embeddings.base import BaseEmbedding
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import numpy as np
from config import EMBED_MODEL_NAME
from typing import Any


class HFEmbedding(BaseEmbedding):
    # Declare the fields so they are recognized
    tokenizer: Any = None
    model: Any = None

    def __init__(self, model_name=EMBED_MODEL_NAME):
        super().__init__()  # Call the parent initializer if needed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _get_text_embedding(self, text: str) -> list:
        text = f"passage: {text.strip()}"
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            # CLS pooling for token embeddings
            embedding = outputs.last_hidden_state[:, 0, :]
        embedding_np = embedding.cpu().numpy()
        normalized_vec = normalize(embedding_np)[0]
        return normalized_vec.tolist()

    def _get_query_embedding(self, text: str) -> list:
        text = f"query: {text.strip()}"
        # Use the same method for query embedding
        return self._get_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> list:
        # For async, simply call the synchronous version
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, text: str) -> list:
        return self._get_query_embedding(text)

    def get_batch_text_embeddings(self, texts: list, batch_size: int = 32) -> list:
        # Batch processing for multiple texts
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [f"passage: {t.strip()}" for t in batch]

            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]  # CLS pooling
                embedding_np = embedding.cpu().numpy()
                normalized_vecs = normalize(embedding_np)
                all_embeddings.extend(normalized_vecs.tolist())

        return all_embeddings
