from indexer import build_index, query_index
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    # Uncomment the next line to build (or rebuild) your FAISS index:
    build_index()


    # Query the system with a sample question:
    question = "What is OpenID Connect and explain its uses for API Gateway?"
    query_index(question)
    question1 = "Provide a sample spec to use this feature with API Gateway"
    query_index(question1)
    question3="Who all are the team members of Ignite?"
    query_index(question3)
