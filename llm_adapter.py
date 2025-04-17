# llm_adapter.py
from llama_index.llms.ollama import Ollama
from config import OLLAMA_API_URL, OLLAMA_MODEL_NAME


def create_ollama_llm():
    """
    Create and return an instance of the official Ollama LLM adapter.
    You can adjust request_timeout (in seconds) if needed.
    """
    return Ollama(
        model=OLLAMA_MODEL_NAME,
        request_timeout=120.0,
        api_url=OLLAMA_API_URL)
