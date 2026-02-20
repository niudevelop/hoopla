import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder

from .keyword_search import InvertedIndex
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        # Pull deep candidate pools
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        # doc_id -> aggregated data (ranks, rrf score, metadata)
        combined: dict[str, dict] = {}

        # BM25 ranks (1-based)
        for rank, r in enumerate(bm25_results, start=1):
            doc_id = r["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": r.get("title"),
                    "document": r.get("document"),
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "score": 0.0,  # RRF score
                }

            # keep best (smallest) rank if duplicates appear
            if (
                combined[doc_id]["bm25_rank"] is None
                or rank < combined[doc_id]["bm25_rank"]
            ):
                combined[doc_id]["bm25_rank"] = rank

            combined[doc_id]["score"] += rrf_score(rank, k)

        # Semantic ranks (1-based)
        for rank, r in enumerate(semantic_results, start=1):
            doc_id = r["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": r.get("title"),
                    "document": r.get("document"),
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "score": 0.0,  # RRF score
                }

            # keep best (smallest) rank if duplicates appear
            if (
                combined[doc_id]["semantic_rank"] is None
                or rank < combined[doc_id]["semantic_rank"]
            ):
                combined[doc_id]["semantic_rank"] = rank

            combined[doc_id]["score"] += rrf_score(rank, k)

        # Sort by fused score desc and return top-k
        fused = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return fused[:limit]


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))

    return normalized_scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def rrf_search_command(
    query: str,
    k: int = DEFAULT_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance_method: str = None,
    rerank_method: str = None,
    evaluate: bool = False,
) -> dict:
    scores = None
    movies = load_movies()
    searcher = HybridSearch(movies)

    if enhance_method:
        query = llm_enahnce_check(query, enhance_method)
    original_query = query

    search_limit = limit
    if rerank_method == "individual":
        search_limit *= 5
    results = searcher.rrf_search(query, k, search_limit)

    if rerank_method == "individual":
        for idx, doc in enumerate(results):
            doc["rerank"] = llm_rerank(query, doc, rerank_method)
            results[idx] = doc
            time.sleep(3)
    elif rerank_method == "batch":
        results = llm_rerank_batch(query, results)
    elif rerank_method == "cross_encoder":
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        pairs = []
        for idx, doc in enumerate(results):
            pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

        scores = cross_encoder.predict(pairs)
        for doc, score in zip(results, scores):
            doc["rerank"] = float(score)

        results.sort(key=lambda d: d["rerank"], reverse=True)

    if evaluate:
        scores = llm_score_results(query, results)

    return {
        "original_query": original_query,
        "query": query,
        "k": k,
        "results": results[:limit],
        "scores": scores,
    }


def llm_enahnce_check(query: str, method: str) -> str:
    client = genai.Client(api_key=api_key)
    match method:
        case "spell":
            prompt = f"""Fix any spelling errors in this movie search query.
Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
        case "rewrite":
            prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
        case "expand":
            prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

        case _:
            return query

    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
        # config=types.GenerateContentConfig(
        #     system_instruction=
        # )
    )
    enhanced_query = resp.text
    print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'")

    return enhanced_query


def llm_rerank(query: str, doc, method: str) -> str:
    client = genai.Client(api_key=api_key)
    match method:
        case "individual":
            prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text, linebreaks, formatting or explanation.

Score:"""

        case _:
            return query

    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
        # config=types.GenerateContentConfig(
        #     system_instruction=
        # )
    )
    # enhanced_query = resp.text
    # print(f"Enhanced query (spell): '{query}' -> '{enhanced_query}'")

    return int(resp.text.replace("\n", ""))


def llm_rerank_batch(query: str, docs: list[dict]) -> list[dict]:

    client = genai.Client(api_key=api_key)

    for d in docs:
        if "id" not in d:
            raise KeyError("Each doc must include an integer 'id' for batch rerank.")

    doc_list_str = "\n".join(
        f'ID: {d["id"]} | Title: {d.get("title","")} | Description: {d.get("document","")}'
        for d in docs
    )

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )

    raw = (resp.text or "").strip()

    try:
        ranked_ids = json.loads(raw)
        if not isinstance(ranked_ids, list):
            raise ValueError("Batch rerank response was not a JSON list.")
    except Exception:
        for i, d in enumerate(docs, start=1):
            d["rerank"] = i
        return docs

    by_id = {d["id"]: d for d in docs}
    seen = set()

    reranked = []
    rank = 1
    for _id in ranked_ids:
        if _id in by_id and _id not in seen:
            d = by_id[_id]
            d["rerank"] = rank
            reranked.append(d)
            seen.add(_id)
            rank += 1

    for d in docs:
        if d["id"] not in seen:
            d["rerank"] = rank
            reranked.append(d)
            rank += 1

    return reranked


def build_llm_eval_prompt(query: str, results: list[dict]) -> str:
    formatted_results = []
    for i, d in enumerate(results, start=1):
        title = d.get("title", "")
        desc = d.get("document", "")
        formatted_results.append(f"{i}. {title} - {desc}")

    return f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""


def llm_score_results(query: str, results: list[dict]) -> list[int]:
    client = genai.Client(api_key=api_key)

    prompt = build_llm_eval_prompt(query, results)
    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )

    raw = (resp.text or "").strip()

    try:
        scores = json.loads(raw)
        if not isinstance(scores, list):
            raise ValueError("LLM eval response was not a JSON list.")
        if len(scores) != len(results):
            raise ValueError("LLM eval response length mismatch.")
        for s in scores:
            if s not in (0, 1, 2, 3):
                raise ValueError("LLM eval response contained out-of-range score.")
        return scores
    except Exception:
        # hard-fail safe: mark everything not relevant
        return [0] * len(results)
