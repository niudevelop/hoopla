import argparse

from lib.hybrid_search import (
    normalize_scores,
    rrf_search_command,
    weighted_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )
    rrf_parser = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("--k", default=60, type=int, help="Top k")
    rrf_parser.add_argument("--evaluate", action="store_true", help="Evaluate")
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Rerank method",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
        case "rrf-search":
            result = rrf_search_command(
                args.query,
                args.k,
                args.limit,
                args.enhance,
                args.rerank_method,
                args.evaluate,
            )
            if args.rerank_method:
                print(
                    f"Reranking top {args.limit} results using {args.rerank_method} method..."
                )
                print(
                    f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):"
                )
            else:
                print(
                    f"RRF Hybrid Search Results for '{result['query']}' (k={result['k']}):"
                )
                print(
                    f"  K {result['k']}: {int(result['k'])}% Keyword, {int((1 - result['k']))}% Semantic"
                )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if args.rerank_method == "individual" or args.rerank_method == "batch":
                    print(f"   Rerank Score: {res.get('rerank', 0):.3f}/10")
                elif args.rerank_method == "cross_encoder":
                    print(f"   Cross Encoder Score: {res.get('rerank', 0):.3f}/10")

                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25 Rank: {metadata['bm25_score']:.3f}, Semantic Rank: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")

                print()

            if args.evaluate:
                res = result["results"]
                scores = result["scores"]
                for i, (doc, score) in enumerate(zip(res, scores), start=1):
                    title = doc.get("title", "")
                    print(f"{i}. {title}: {score}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
