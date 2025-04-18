from indexer import build_index, query_index
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    # Uncomment the next line to build (or rebuild) your FAISS index:
    build_index()


    # Query the system with a sample question:
    question = "What is OpenID Connect and explain its uses for API Gateway?"
    #query_index(question)
    print("======================================================================")
    question2 = "What is response caching and how can we do that in API Gateway?"
    query_index(question2)
    question2 = "What is response caching and how can we do that in API Gateway?"
    query_index(question2)
    question2 = "What is response caching and how can we do that in API Gateway?"
    query_index(question2)
    question2 = "What is response caching and how can we do that in API Gateway?"
    query_index(question2)
    print("======================================================================")
    question3="Who all are the team members of Ignite?"
    query_index(question3)