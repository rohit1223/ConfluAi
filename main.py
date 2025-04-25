import argparse
import multiprocessing
from indexer import build_index, query_index

DEFAULT_QUESTIONS = [
    "How do I set up custom domains and tls certificates with api gateway?",
    "How can I support multi-tenant token validation with dynamic auth routing?",
    "What is the purpose of the responsecachelookup and responsecachestorage policies in the API deployment JSON?",
    "Can you provide a JSON example where an API deployment has multiple routes and applies different backend for each routes?",
    "How can I upload an API description file for use in creating an API deployment specification?",
    "Who all are the team members of Ignite?"
]

def main():
    parser = argparse.ArgumentParser(
        description="Build/rebuild FAISS index and/or query it from the command line"
    )
    parser.add_argument(
        "-b", "--build-index",
        action="store_true",
        help="Build (or rebuild) the FAISS index before querying"
    )
    parser.add_argument(
        "-B", "--build-only",
        action="store_true",
        help="After building the index, exit immediately without running any queries"
    )
    parser.add_argument(
        "-q", "--query",
        action="append",
        metavar="QUESTION",
        help="Question to ask the index. Can be specified multiple times."
    )

    args = parser.parse_args()

    # Safe multiprocessing setup on Mac/Linux
    multiprocessing.set_start_method("spawn", force=True)

    # Build or rebuild the index if requested
    if args.build_index:
        print("[*] Building FAISS index…")
        build_index()
        print("[✓] Index built successfully.\n")

    # If user explicitly asked for build-only, exit now
    if args.build_only:
        print("[*] Build-only flag set; skipping all queries.")
        return

    # Choose which questions to run
    questions = args.query if args.query else DEFAULT_QUESTIONS

    # Run queries
    for q in questions:
        print(f"Query: {q!r}")
        query_index(q)
        print("=" * 70)

if __name__ == "__main__":
    main()
