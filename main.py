from indexer import build_index, query_index
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    # Uncomment the next line to build (or rebuild) your FAISS index:
    build_index()


    # Query the system with a sample question:
    question1 = "How would I pass the client certificate in a custom header to the backend?"
    query_index(question1)
    print("======================================================================")
    question2 = "How can I support multi-tenant token validation with dynamic auth routing?"
    query_index(question2)
    print("======================================================================")
    question3 = "What is the purpose of the responsecachelookup and responsecachestorage policies in the API deployment JSON?"
    query_index(question3)
    print("======================================================================")
    question4 = "Can you provide a JSON example where an API deployment has multiple routes and applies different request policies to each?"
    query_index(question4)
    print("======================================================================")
    question5 = "How can I upload an API description file for use in creating an API deployment specification?"
    query_index(question5)
    print("======================================================================")
    question6="Who all are the team members of Ignite?"
    query_index(question6)
    print("======================================================================")