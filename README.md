# ConfluAI: A Local RAG System with LLaMA 3.2, Hugging Face Embeddings, and FAISS

ConfluAI is a Retrieval-Augmented Generation (RAG) system built to run entirely locally. It leverages modern NLP technologies to answer questions and engage in continuous chat sessions using your project documentation (e.g., Confluence exports) as the knowledge base.

## Features

- **Local Processing:** All components run on your local machine—no external API keys or cloud dependencies.
- **Web Scraping:** Unintigrated webscraper snippet to retrieve confluence content. 
- **Embeddings:** Uses Hugging Face Transformers (e.g., `BAAI/bge-small-en-v1.5`) to convert text into dense vector representations.
- **Vector Retrieval:** Utilizes FAISS for fast and efficient vector search.
- **LLM Integration:** Queries a local LLaMA 3.2 model served via Ollama.
- **Context-aware Chat:** Supports context-aware quention queries.
- **Modular Design:** Clear separation of concerns across multiple modules for ease of maintenance and extension.

### WIP Feature
- Integrated web scraper
- Continuous conversations using a custom workflow

## Repository Structure
```bash
   ConfluAi/
   ├── README.md              # This file – repository overview and instructions
   ├── base_urls.txt          # A text file with the list of base URLs for web scraping
   ├── config.py              # Global configuration file (paths, model names, etc.)
   ├── confluence_docs/       # Directory to store scraped and cleaned Confluence documents (as text files)
   ├── embeddings.py          # Module that implements the HFEmbedding adapter using Hugging Face Transformers
   ├── faiss_index/           # Directory where the FAISS vector store and associated JSON files are persisted
   ├── indexer.py             # Core module for building the FAISS index and querying it with LlamaIndex
   ├── llm_adapter.py         # LLM adapter for querying the local LLaMA 3.2 via Ollama
   ├── scrapper.py            # Web scraper module for collecting Confluence docs (moved to root)
   ├── main.py                # Main entry point for ingesting documents and querying the system
   ├── requirements.txt       # List of required Python packages
   ├── tests/                 # Directory containing test scripts for the project
   │   ├── test_embedding.py  # Test file for embeddings functionality
   │   └── test_fiass.py      # Test file for FAISS-related functionality
   └── utils/                 # Directory for additional utilities and helper functions
      └── utils.py           # Other helper functions (e.g., cleaning, file operations)

```


## Getting Started

### Prerequisites

- **Python 3.9.6+**
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
3. **Create configuration files and directory**
   ```bash
   mkdir confluence_docs # Should match the DOCS_DIR in the config.py
   touch base_urls.txt 
   ```
4. **Populate the `base_urls.txt` with scraping URLs**
   Example:
      ```python
      https://example.com/
      https://dummyjson.com/docs
      https://www.postman.com/api-platform/api-testing/
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

VALID_LINK_SUBSTRING = "Content/<YourContent>"
VALID_LINK_EXTENSION = ".htm"
EXCLUDE_EXTENSIONS = ['.pdf', '.zip']
```

### Ingesting Documents and Building the Index
1. **Prepare Your Data:**
   1. Create the directoy specified in `DOCS_DIR` in the `config.py` Place your cleaned Confluence documents (plain text or markdown) in the directory specified by `DOCS_DIR`.
   2. Create a file called `base_urls.txt` and populate it with the list of urls to scrape.
   3. `scrapper.py` is an standalone modeul that has to be run separately to populate `DOCS_DIR`.
      ```bash
      python3 scrapper.py
      ```

2. **Build the Index:**
In main.py, uncomment the build_index() call and run:
    ```bash
    python3 main.py
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
    python3 main.py 
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
