import os
import pickle
import json
import math
from collections import Counter
from lib.refine_query import refine_query

class InvertedIndex:
    index = {}
    docmap = {}
    term_frequencies = {}
    def __add_document(self, doc_id, text):
        tokens = refine_query(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        for t in tokens:
            if t not in self.index:
                self.index[t] = set()
            self.index[t].add(doc_id)
    def get_documents(self, term):
        ids = list(self.index[term.lower()])
        return sorted(ids)
    def build(self):
        with open("data/movies.json") as f:
            data = json.load(f)
            for movie in data["movies"]:
                self.docmap[movie["id"]] = movie
                mstr = movie["title"] + " " + movie["description"]
                self.__add_document(movie["id"], mstr)
    def save(self):
        path = "cache/"
        if not os.path.exists(path):
            os.makedirs(path)
        with open("cache/index.pkl", "wb") as i:
            pickle.dump(self.index, i)
        with open("cache/docmap.pkl", "wb") as d:
            pickle.dump(self.docmap, d)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
    def load(self):
        if os.path.isfile("cache/index.pkl"):
            with open("cache/index.pkl", "rb") as i:
                self.index = pickle.load(i)
        else:
            raise Exception("index.pkl not found")
        if os.path.isfile("cache/docmap.pkl"):
            with open("cache/docmap.pkl", "rb") as d:
                self.docmap = pickle.load(d)
        else:
            raise Exception("docmap.pkl not found")
        if os.path.isfile("cache/term_frequencies.pkl"):
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
        else:
            raise Exception("term_frequencies.pkl not found")
    def get_tf(self, doc_id, term):
        if term not in self.term_frequencies[doc_id]:
            return 0
        return self.term_frequencies[doc_id][term]
    def get_idf(self, term):
        doc_count = len(self.docmap)
        term_doc_count = 0
        if term in self.index:
            term_doc_count = len(self.index[term])
        return math.log((doc_count + 1) / (term_doc_count + 1))