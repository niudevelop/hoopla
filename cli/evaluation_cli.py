import argparse
import json

from lib.search_utils import DATA_PATH, EVAL_PATH
from lib.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    k = args.k

    # run evaluation logic here
    with open(EVAL_PATH, "r") as f:
        test_cases = json.load(f)["test_cases"]

    results = []

    for test in test_cases:
        query = test["query"]
        relevant_docs = test.get("relevant_docs", [])

        out = rrf_search_command(
            query=query,
            k=60,
            limit=limit,
        )

        retrieved_titles = [doc["title"] for doc in out["results"]]
        precision = precision_at_k(retrieved_titles, relevant_docs)
        recall = recall_at_k(retrieved_titles, relevant_docs)
        f1 = 2 * (precision * recall) / (precision + recall)

        results.append(
            {
                "query": query,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "retrieved": retrieved_titles,
                "relevant": relevant_docs,
            }
        )

    # printing
    print(f"k={args.limit}\n")

    for r in results:
        print(f"- Query: {r['query']}")
        print(f"  - Precision@{args.limit}: {r['precision']:.4f}")
        print(f"  - Recall@{args.limit}: {r['recall']:.4f}")
        print(f"  - F1 Score: {r['f1']:.4f}")
        print(f"  - Retrieved: {', '.join(r['retrieved'])}")
        print(f"  - Relevant: {', '.join(r['relevant'])}\n")


def precision_at_k(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    if not retrieved_titles:
        return 0.0

    relevant_set = set(relevant_titles)
    relevant_retrieved = sum(1 for title in retrieved_titles if title in relevant_set)

    return relevant_retrieved / len(retrieved_titles)


def recall_at_k(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    if not relevant_titles:
        return 0.0
    retrieved = set(retrieved_titles)
    hits = sum(1 for t in set(relevant_titles) if t in retrieved)
    return hits / len(set(relevant_titles))


if __name__ == "__main__":
    main()
