from llama_index.core.prompts import RichPromptTemplate

# No‑hallucination prompt template: only answer from provided context, otherwise say "I don't know."
NO_HALLU_TEMPLATE = RichPromptTemplate(
    template_str=(
        "<<SYSTEM>>\n"
        "You are a knowledge assistant.\n"
        "Use ONLY the provided CONTEXT to answer the QUESTION.\n"
        "If the answer is not in CONTEXT, reply exactly \"I don't know.\"\n\n"
        "=== CONTEXT ===\n"
        "{context_str}\n\n"
        "=== QUESTION ===\n"
        "{query_str}\n\n"
        "=== ANSWER ===\n"
    ),
    input_variables=["context_str", "query_str"],
)
