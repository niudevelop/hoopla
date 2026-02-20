from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.search_utils import load_movies
from lib.semantic_search import cosine_similarity


class MultimodalSearch:

    def __init__(
        self,
        documents: list,
        model_name="clip-ViT-B-32",
    ):
        self.model = SentenceTransformer(model_name)
        self.docs = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.docs]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, path: str):
        image = Image.open(path)
        self.embeddings = self.model.encode([image])[0]
        return self.embeddings

    def search_with_image(self, path: str):
        image_embedding = self.embed_image(path)
        results = []
        for doc, text_embedding in zip(self.docs, self.text_embeddings):
            score = cosine_similarity(text_embedding, image_embedding)
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity": float(score),
                }
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]


def verify_image_embedding(path: str):
    mm = MultimodalSearch()
    embedding = mm.embed_image(path)
    return embedding


def image_search_command(path):
    movies = load_movies()
    mm = MultimodalSearch(movies)
    return mm.search_with_image(path)
