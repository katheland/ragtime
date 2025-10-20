#!/usr/bin/env python3

import argparse
import json
import string
from lib.inverted_index import InvertedIndex
from lib.refine_query import refine_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the Inverted Index")

    tf_parser = subparsers.add_parser("tf", help="in progress")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="in progress")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="in progress")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="term")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for: " + args.query)
            q = refine_query(args.query)
            idx = InvertedIndex()
            try:
                idx.load()
                results = set()
                for token in q:
                    if token in idx.index:
                        results = results.union(idx.index[token])
                    if len(results) >= 5:
                        break
                sorted_results = sorted(list(results))
                for i in range(5):
                    if i < len(sorted_results):
                        doc_id = sorted_results[i]
                        print(f"{doc_id}: {idx.docmap[doc_id]["title"]}")
            except Exception as e:
                print(e)
        case "build":
            print("build in progress")
            idx = InvertedIndex()
            idx.build()
            idx.save()
            print("build complete")
        case "tf":
            idx = InvertedIndex()
            refined = refine_query(args.term)
            if len(refined) != 1:
                print(f"the get_tf query {args.term} should be one word long")
                return
            t = refined[0]
            try:
                idx.load()
                print(f"{args.term} in {args.doc_id}: {idx.get_tf(args.doc_id, t)}")
            except Exception as e:
                print(e)
        case "idf":
            idx = InvertedIndex()
            refined = refine_query(args.term)
            if len(refined) != 1:
                print(f"the get_tf query {args.term} should be one word long")
                return
            t = refined[0]
            try:
                idx.load()
                idf = idx.get_idf(t)
                print(f"{doc_count} {term_doc_count} {idf}")
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            except Exception as e:
                print(e)
        case "tfidf":
            idx = InvertedIndex()
            refined = refine_query(args.term)
            if len(refined) != 1:
                print(f"the get_tf query {args.term} should be one word long")
                return
            t = refined[0]
            try:
                idx.load()
                tf = idx.get_tf(args.doc_id, t)
                idf = idx.get_idf(t)
                tf_idf = tf * idf
                print(f"{tf} {idf} {tf_idf}")
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            except Exception as e:
                print(e)
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()