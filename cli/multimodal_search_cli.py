import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify Image Embeddings"
    )
    verify_parser.add_argument("image_path", help="Path to image")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Verify Image Embeddings"
    )
    image_search_parser.add_argument("image_path", help="Path to image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            path = args.image_path
            embedding = verify_image_embedding(path)
            print(f"Embedding shape: {embedding.shape[0]} dimensions")
        case "image_search":
            path = args.image_path
            docs = image_search_command(path)

            for i, doc in enumerate(docs, start=1):
                print(f"{i}. {doc["title"]} (similarity: {doc["similarity"]:.3f})")
                print(f"    {doc["description"]}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
