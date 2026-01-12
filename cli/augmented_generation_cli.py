import argparse

from lib.augmented_generation import question, rag_search, summarize, citations
from lib.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser("summarize", help="Summarize results")
    summarize_parser.add_argument(
        "query", type=str, help="Search query for Summarization"
    )
    summarize_parser.add_argument(
        "--limit", default=5, type=int, help="Number of results to summarize"
    )
    citations_parser = subparsers.add_parser("citations", help="Citate results")
    citations_parser.add_argument("query", type=str, help="Search query for Rag")
    citations_parser.add_argument(
        "--limit", default=5, type=int, help="Number of results"
    )

    question_parser = subparsers.add_parser("question", help="Question Answering")
    question_parser.add_argument("question", type=str, help="Question you want to ask")
    question_parser.add_argument(
        "--limit", default=5, type=int, help="Number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rrf_search_command(query)
            docs = results["results"]
            resp = rag_search(query, docs)
            print(f"Search Results:")
            for doc in docs:
                print(f"  -{doc["title"]}")
            print()
            print(f"RAG Response:")
            print(resp)
        case "summarize":
            query = args.query
            limit = args.limit
            results = rrf_search_command(query, limit=limit)
            docs = results["results"]
            resp = summarize(query, docs)
            print(f"Search Results:")
            for doc in docs:
                print(f"  -{doc["title"]}")
            print()
            print(f"RAG Response:")
            print(resp)
        case "citations":
            query = args.query
            limit = args.limit
            results = rrf_search_command(query, limit=limit)
            docs = results["results"]
            resp = citations(query, docs)
            print(f"Search Results:")
            for doc in docs:
                print(f"  -{doc["title"]}")
            print()
            print(f"LLM Answer:")
            print(resp)
        case "question":
            query = args.question
            limit = args.limit
            results = rrf_search_command(query, limit=limit)
            docs = results["results"]
            resp = question(query, docs)
            print(f"Search Results:")
            for doc in docs:
                print(f"  -{doc["title"]}")
            print()
            print(f"Answer:")
            print(resp)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
