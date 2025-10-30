from sentence_transformers import SentenceTransformer
import numpy as np
import os
import operator

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")
    def generate_embedding(self, text):
        if text == "" or text.strip() == "":
            raise ValueError("You need to pass some actual text.")
        text_list = [text]
        embedding = self.model.encode(text_list)
        return embedding[0]
    def build_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar = True)
        if not os.path.exists("cache/"):
            os.makedirs("cache/")
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded.  Call 'load_or_create_embeddings' first.")
        query_embedding = self.generate_embedding(query)
        comparisons = []
        for i in range(len(self.embeddings)):
            similarity = cosine_similarity(query_embedding, self.embeddings[i])
            comparisons.append((similarity, self.documents[i]))
        #print(comparisons[0])
        sort_comparisons = sorted(comparisons, key=operator.itemgetter(0), reverse=True)
        result = []
        for j in range(limit):
            entry = {
                "score": sort_comparisons[j][0],
                "title": sort_comparisons[j][1]["title"],
                "description": sort_comparisons[j][1]["description"]
            }
            result.append(entry)
        return result

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)