#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verify model")
    embed_parser = subparsers.add_parser("embed_text", help="Verify model")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
