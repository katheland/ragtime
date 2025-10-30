#!/usr/bin/env python3

import argparse
import json
import string
from lib.inverted_index import InvertedIndex
from lib.refine_query import refine_query
from lib.constants import *

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

    bm25idf_parser = subparsers.add_parser("bm25idf", help="in progress")
    bm25idf_parser.add_argument("term", type=str, help="term")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="in progress")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="term")
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=DEFAULT_K1, help="K1 value")
    bm25tf_parser.add_argument("b", type=float, nargs="?", default=DEFAULT_B, help="B value")

    bm25search_parser = subparsers.add_parser("bm25search", help="in progress")
    bm25search_parser.add_argument("query", type=str, help="Search Query")
    bm25search_parser.add_argument("limit", type=int, nargs="?", default=RESULTS_LENGTH, help="search limit")
    
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
                    if len(results) >= RESULTS_LENGTH:
                        break
                sorted_results = sorted(list(results))
                for i in range(RESULTS_LENGTH):
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
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            except Exception as e:
                print(e)
        case "bm25idf":
            idx = InvertedIndex()
            refined = refine_query(args.term)
            if len(refined) != 1:
                print(f"the get_tf query {args.term} should be one word long")
                return
            t = refined[0]
            try:
                idx.load()
                bm25idf = idx.get_bm25_idf(t)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except Exception as e:
                print(e)
        case "bm25tf":
            idx = InvertedIndex()
            refined = refine_query(args.term)
            if len(refined) != 1:
                print(f"the get_tf query {args.term} should be one word long")
                return
            t = refined[0]
            try:
                idx.load()
                bm25tf = idx.get_bm25_tf(args.doc_id, t, args.k1, args.b)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except Exception as e:
                print(e)
        case "bm25search":
            idx = InvertedIndex()
            try:
                idx.load()
                results = idx.bm25_search(args.query, args.limit)
                for r in results:
                    print(f"({r}) {idx.docmap[r]["title"]} - Score: {results[r]:.2f}")
            except Exception as e:
                print(e)
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()