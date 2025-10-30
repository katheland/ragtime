#!/usr/bin/env python3

import argparse
from lib.semantic_search import *
import json
from lib.constants import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Search movies using BM25")

    embedtext_parser = subparsers.add_parser("embed_text", help="embed text")
    embedtext_parser.add_argument("text", type=str, help="Text")  

    embedquery_parser = subparsers.add_parser("embedquery", help="embed query")
    embedquery_parser.add_argument("query", type=str, help="Query")

    verifyembeddings_parser = subparsers.add_parser("verify_embeddings", help="verify embeddings")

    search_parser = subparsers.add_parser("search", help="search")
    search_parser.add_argument("query", type=str, help="Query")
    search_parser.add_argument("--limit", type=int, nargs="?", default=RESULTS_LENGTH, help="search limit")
    
    args = parser.parse_args()
        
    match args.command:
        case "verify":
            smsch = SemanticSearch()
            smsch.verify_model()
        case "embed_text":
            smsch = SemanticSearch()
            embedding = smsch.generate_embedding(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")
        case "verify_embeddings":
            smsch = SemanticSearch()
            with open("data/movies.json") as f:
                data = json.load(f)
                documents = data["movies"]
                embeddings = smsch.load_or_create_embeddings(documents)
                print(f"Number of docs:   {len(documents)}")
                print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
        case "embedquery":
            smsch = SemanticSearch()
            embedding = smsch.generate_embedding(args.query)
            print(f"Query: {args.query}")
            print(f"First 5 dimensions: {embedding[:5]}")
            print(f"Shape: {embedding.shape}")
        case "search":
            smsch = SemanticSearch()
            with open("data/movies.json") as f:
                data = json.load(f)
                documents = data["movies"]
                embeddings = smsch.load_or_create_embeddings(documents)
                result = smsch.search(args.query, args.limit)
                for entry in result:
                    print(f"{entry['title']} (score: {entry['score']})")
                    print(entry['description'])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()