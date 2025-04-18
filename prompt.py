from llama_index.core.prompts import RichPromptTemplate

# No‑hallucination prompt template: only answer from provided context, otherwise say "I don't know."
NO_HALLU_TEMPLATE = RichPromptTemplate(
    """
<<SYSTEM>>
You are a knowledge assistant.
Use ONLY the provided CONTEXT to answer the QUESTION.
If the answer is not in CONTEXT, reply exactly "I don't know."

=== CONTEXT ===
{{ context_str }}

=== QUESTION ===
{{ query_str }}

=== ANSWER ===
"""
)