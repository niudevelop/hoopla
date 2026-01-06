import os
import json


def load_movies():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, "../data", "movies.json")
    with open(movies_path, "r") as f:
        return json.load(f)["movies"]
