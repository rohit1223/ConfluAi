# ConfluAI: A Local RAG System with LLaMA 3.2, Hugging Face Embeddings, and FAISS

ConfluAI is a Retrieval-Augmented Generation (RAG) system built to run entirely locally. It leverages modern NLP technologies to answer questions and engage in continuous chat sessions using your project documentation (e.g., Confluence exports) as the knowledge base.

## Features

- **Local Processing:** All components run on your local machine—no external API keys or cloud dependencies.
- **Embeddings:** Uses Hugging Face Transformers (e.g., `BAAI/bge-small-en-v1.5`) to convert text into dense vector representations.
- **Vector Retrieval:** Utilizes FAISS for fast and efficient vector search.
- **LLM Integration:** Queries a local LLaMA 3.2 model served via Ollama.
- **P1+r4632=1B5B32347E\P0+r\P0+r\P1+r6B62=7F\P0+r\P1+r6B44=1B5B337E\P1+r6B68=1B4F48\P1+r4037=1B4F46\P1+r6B50=1B5B357E\P1+r6B4E=1B5B367E\Continuous Chat:** Supports context-aware, continuous conversations using a custom workflow.
- **Modular Design:** Clear separation of concerns across multiple modules for ease of maintenance and extension.

## Repository Structure
```bash
    ConfluAI/
    ├── config.py         # Global configuration (directories, model names, etc.)
    ├── embeddings.py     # Hugging Face embedding adapter (HFEmbedding)
    ├── llm_adapter.py    # LLM adapter using the official Ollama adapter for LLaMA 3.2
    ├── indexer.py        # Functions to build and query the FAISS index with LlamaIndex
    ├── utils.py          # Utility functions (document loading, cleaning, chunking)
    ├── main.py           # Main entry point to build and query the index
    └── chat.py           # Continuous chat example with context using a custom workflow
```


## Getting Started

### Prerequisites

- **Python 3.9+**
- **Ollama:** Install and run [Ollama](https://ollama.com/) on your local machine to serve your LLaMA 3.2 model.
- **Required Python Packages:** See the Installation section below.

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ConfluAI.git
   cd ConfluAI
2. **Create a Virtual Environment and Activate It:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
### Configuration

Edit `config.py` to configure your paths and models. For example:

```python
# config.py
DOCS_DIR = "./confluence_docs"         # Directory with your cleaned Confluence documents (.txt/.md)
FAISS_INDEX_PATH = "faiss_index"         # Directory for persisting the FAISS index and associated JSON files
EMBEDDING_DIM = 384                      # Embedding dimension (adjust based on your model)

OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint
OLLAMA_MODEL_NAME = "llama3.2:latest"      # Specify your local LLaMA model (adjust as needed)
```

### Ingesting Documents and Building the Index
1. **Prepare Your Data:**
Place your cleaned Confluence documents (plain text or markdown) in the directory specified by DOCS_DIR.

2. **Build the Index:**
In main.py, uncomment the build_index() call and run:
    ```bash
    python main.py
    ```

    This will:
   1. Load, clean, and chunk your documents.

   2. Compute embeddings using your Hugging Face model.

   3. Build a FAISS index with LlamaIndex.
 
   4. Persist the index (including docstore, vector store, and metadata) to the directory specified by FAISS_INDEX_PATH.

3. **Querying the Index**

    After building the index:
   1. Disable Index Building:
      Comment out or remove the build_index() call in main.py.

   2. Query the System:
      Modify the sample question if needed, then run:
    ```bash
    python main.py 
    ```
   This loads the persisted index, retrieves relevant context, assembles a prompt, and uses your local LLaMA 3.2 (via Ollama) to generate an answer.

### Troubleshooting

1. **Persisted Files Missing:**
   Ensure that you have run the indexing phase successfully. Your persist directory `(FAISS_INDEX_PATH)` should contain several JSON files such as `default__vector_store.json`, `docstore.json`, and `index_store.json`.

2. **Ollama Issues:**
       Verify that your Ollama instance is running on `http://localhost:11434` and that your specified model is loaded.

3. **LLM API Key Errors:**
Since this project uses a local LLM via Ollama, no external API keys (like OpenAI’s) are required. If you see errors related to API keys, ensure your configuration in `config.py` is correct.

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or new features.

### License
This project is licensed under the MIT License.
