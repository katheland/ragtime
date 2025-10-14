#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for: " + args.query)
            s = open("data/stopwords.txt")
            stop_words = s.read().splitlines()
            table = str.maketrans("", "", string.punctuation)
            stemmer = PorterStemmer()
            with open("data/movies.json") as f:
                data = json.load(f)
                results = []
                q = refine_query(args.query, stop_words, stemmer)
                for movie in data["movies"]:
                    append = False
                    m = refine_query(movie["title"], stop_words, stemmer)
                    for i in range(len(q)):
                        for j in range(len(m)):
                            if q[i].translate(table) in m[j].translate(table):
                                append = True
                    if append == True:
                        results.append(movie["title"])    
                found = results[:5]
                for i in range(len(found)):
                    print(str(i+1) + ": " + found[i])
        case _:
            parser.print_help()

def refine_query(query_string, stop_words, stemmer):
    q = query_string.lower().split(" ")
    while "" in q:
        q.remove("")
    for word in q:
        if word in stop_words:
            q.remove(word)
    for i in range(len(q)):
        q[i] = stemmer.stem(q[i])
    return q


if __name__ == "__main__":
    main()