# llm_adapter.py
from llama_index.llms.ollama import Ollama
from config import OLLAMA_API_URL, OLLAMA_MODEL_NAME


def create_ollama_llm():
    """
    Create and return an instance of the official Ollama LLM adapter.
    You can adjust request_timeout (in seconds) if needed.
    Also, adjust the temperature and top_p to achieve optimal response.
    """
    return Ollama(
        model=OLLAMA_MODEL_NAME,
        api_url=OLLAMA_API_URL,
        request_timeout=120.0,
        # Increase T if the model repeating the same phrase over and over
        # Decrease T if the model is hallucinating and the response is erratic
        temperature=0.3, 
        # Nucleus Sampling
        # Instead of looking at all tokens, you sort tokens by probability,
        # then take the smallest set whose cumulative probability ≥ 𝑝,
        # and sample from that set uniformly (after re‑normalizing).
        top_p=0.9)
        
