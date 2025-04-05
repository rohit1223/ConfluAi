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
        # Tokenize the input text
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over token embeddings
            embedding = outputs.last_hidden_state.mean(dim=1)
        embedding_np = embedding.cpu().numpy()
        normalized_vec = normalize(embedding_np)[0]
        return normalized_vec.tolist()

    def _get_query_embedding(self, text: str) -> list:
        # Use the same method for query embedding
        return self._get_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> list:
        # For async, simply call the synchronous version
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, text: str) -> list:
        return self._get_query_embedding(text)

    def get_batch_text_embeddings(self, texts: list) -> list:
        # Batch processing for multiple texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings_np = embeddings.cpu().numpy()
        normalized_vecs = normalize(embeddings_np)
        return normalized_vecs.tolist()
