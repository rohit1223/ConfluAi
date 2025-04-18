from llama_index.core.prompts import RichPromptTemplate

NO_HALLU_TEMPLATE = RichPromptTemplate(
    input_variables=["context_str", "query_str"],
    system_prompt=(
        "You are a knowledge assistant for API Gateway Service.  \n"
        "Use ONLY the provided CONTEXT to answer the QUESTION.  \n"
        "If the answer is not in CONTEXT, reply exactly \"I don’t know.\""
    ),
    context_prompt="=== CONTEXT ===\n{context_str}\n",
    question_prompt="=== QUESTION ===\n{query_str}\n",
    answer_prompt="=== ANSWER ===\n"
)
