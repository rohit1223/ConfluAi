# indexer.py
import faiss
import os
import logging
from typing import Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from config import DOCS_DIR, EMBEDDING_DIM, FAISS_INDEX_PATH
from embeddings import HFEmbedding
from llm_adapter import create_ollama_llm
from prompt import NO_HALLU_TEMPLATE
from utils.utils import load_documents_with_metadata, chunk_document

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)


def build_index() -> None:
    """
    Build the FAISS index:
    1. Load documents from DOCS_DIR.
    2. Compute embeddings using HFEmbedding.
    3. Create a FAISS index (using IndexFlatIP) with the specified embedding dimension.
    4. Build a StorageContext with FaissVectorStore.
    5. Construct a VectorStoreIndex using LlamaIndex and persist it.
    """
    logging.info("Clean and loading documents from: %s", DOCS_DIR)

    raw_docs = load_documents_with_metadata(DOCS_DIR)
    documents = []

    for doc in raw_docs:
        text = doc.text
        md   = getattr(doc, "extra_info", {})
        chunks = chunk_document(text)
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                text=chunk,
                metadata={
                    "file_path": md.get("source", md.get("file_path", "unknown")),
                    "chunk_id": i
                }
            )
        )

    logging.info("Initializing Hugging Face embedding model...")
    hf_embedding = HFEmbedding()

    logging.info("Creating FAISS index...")
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    faiss_store = FaissVectorStore(faiss_index=faiss_index)

    logging.info("Building storage context with FAISS vector store...")
    storage_context = StorageContext.from_defaults(vector_store=faiss_store)

    logging.info("Building vector store index from documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=hf_embedding,
        storage_context=storage_context,
        show_progress=True
    )

    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    logging.info("Persisting index")
    index.storage_context.persist(persist_dir=FAISS_INDEX_PATH)
    logging.info("Index built and saved successfully.")


def query_index(question: str) -> None:
    """
    Query the FAISS index:
    1. Initialize HFEmbedding for query-time embedding.
    2. Load the persisted FAISS index.
    3. Reconstruct the VectorStoreIndex from the loaded FaissVectorStore.
    4. Use LlamaIndex's query engine to retrieve the top relevant context.
    5. Assemble a prompt (context + question) and send it to the local LLaMA 3.2 via Ollama.
    6. Print the generated answer.
    """
    logging.info("Initializing Hugging Face embedding model for querying...")
    hf_embedding = HFEmbedding()

    logging.info("Loading FAISS index from")
    faiss_store = FaissVectorStore.from_persist_path(
        "./faiss_index/default__vector_store.json")
    storage_context = StorageContext.from_defaults(
        vector_store=faiss_store, persist_dir=FAISS_INDEX_PATH)

    logging.info("Reconstructing index from FAISS vector store...")
    # index = VectorStoreIndex.from_vector_store(faiss_store, embed_model=hf_embedding)
    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=hf_embedding)

    logging.info("Querying index for relevant context...")
    # query_engine = index.as_query_engine(embed_model=hf_embedding)
    ollama_llm = create_ollama_llm()
    query_engine = index.as_query_engine(
        llm=ollama_llm, 
        embed_model=hf_embedding,
        text_qa_template=NO_HALLU_TEMPLATE,
        similarity_top_k=6,
        verbose=False)

    retrieved_context: Any = query_engine.query(question)
    answer = getattr(retrieved_context, "response", str(retrieved_context))
    source_nodes = getattr(retrieved_context, "source_nodes", None)
    if source_nodes:
        print("\n=== CITATIONS ===")
        for node in source_nodes:
            meta = getattr(node, "source_node", getattr(node, "node", node)).metadata or {}
            file_path = meta.get("file_path", meta.get("source", "<unknown>"))
            chunk_id  = meta.get("chunk_id", "<no-id>")
            print(f"- {file_path} (chunk {chunk_id})")

    context = str(retrieved_context)
    
    # Finally, print the answer
    print("\n=== ANSWER ===")
    print(answer) 