import json
import os
import numpy as np
from lib.search_utils import (
    CACHE_DIR,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    load_movies,
)
from lib.semantic_search import SemanticSearch, cosine_similarity, semantic_chunk


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(
                text,
                max_chunk_size=200,
                overlap=1,
            )

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query)

        chunk_scores: list[dict] = []
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            meta = self.chunk_metadata[i]
            score = cosine_similarity(query_embedding, chunk_emb)
            chunk_scores.append(
                {
                    "chunk_idx": meta["chunk_idx"],
                    "movie_idx": meta["movie_idx"],
                    "score": float(score),
                }
            )

        best_by_movie: dict[int, dict] = {}
        for cs in chunk_scores:
            mi = cs["movie_idx"]
            if mi not in best_by_movie or cs["score"] > best_by_movie[mi]["score"]:
                best_by_movie[mi] = cs

        top = sorted(
            best_by_movie.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:limit]

        results: list[dict] = []
        for item in top:
            doc = self.documents[item["movie_idx"]]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": (doc.get("description", "") or "")[:100],
                    "score": round(item["score"], 4),
                    "metadata": {
                        "movie_idx": item["movie_idx"],
                        "chunk_idx": item["chunk_idx"],
                    },
                }
            )

        return results


def embed_chunks():
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {embeddings} chunked embeddings")


def search_chunk(query: str, limit: int = 5):
    searcher = ChunkedSemanticSearch()
    documents = load_movies()
    searcher.load_or_create_chunk_embeddings(documents)
    results = searcher.search_chunks(query, limit)

    for i, doc in enumerate(results, start=1):
        print(
            f"{i}. {doc['title']} (score: {doc['score']:.4f})\n"
            f"   {doc['document']}...\n"
        )
