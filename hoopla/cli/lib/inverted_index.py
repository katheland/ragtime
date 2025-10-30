import os
import pickle
import json
import math
from collections import Counter
from lib.refine_query import refine_query
from lib.constants import *

class InvertedIndex:
    index = {}
    docmap = {}
    term_frequencies = {}
    doc_lengths = {}
    
    def __add_document(self, doc_id, text):
        tokens = refine_query(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id] = Counter(tokens)
        for t in tokens:
            if t not in self.index:
                self.index[t] = set()
            self.index[t].add(doc_id)
    
    def __get_avg_doc_length(self):
        if len(self.doc_lengths) == 0:
            return 0.0
        num_docs = len(self.doc_lengths)
        num_words = 0
        for doc in self.doc_lengths:
            num_words += self.doc_lengths[doc]
        return num_words / num_docs
    
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
        with open("cache/doc_lengths.pkl", "wb") as l:
            pickle.dump(self.doc_lengths, l)
    
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
        if os.path.isfile("cache/doc_lengths.pkl"):
            with open("cache/doc_lengths.pkl", "rb") as l:
                self.doc_lengths = pickle.load(l)
        else:
            raise Exception("doc_lengths.pkl not found")
    
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
    
    def get_bm25_idf(self, term):
        doc_count = len(self.docmap)
        term_doc_count = 0
        if term in self.index:
            term_doc_count = len(self.index[term])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=DEFAULT_K1, b=DEFAULT_B):
        raw_tf = self.get_tf(doc_id, term)
        avg_len = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_len)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def get_bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit):
        tokens = refine_query(query)
        scores = {}
        for t in tokens:
            if t in self.index:
                for doc in self.index[t]:
                    if doc not in scores:
                        scores[doc] = 0.00
                    scores[doc] += self.get_bm25(doc, t)
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit])
        return sorted_scores