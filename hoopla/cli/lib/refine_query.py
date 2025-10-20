from nltk.stem import PorterStemmer
import string

def refine_query(query_string):
    query = query_string.translate(str.maketrans("", "", string.punctuation))
    s = open("data/stopwords.txt")
    stop_words = s.read().splitlines()
    s.close()
    stemmer = PorterStemmer()
    q = query.lower().split(" ")
    while "" in q:
        q.remove("")
    for word in q:
        if word in stop_words:
            q.remove(word)
    for i in range(len(q)):
        q[i] = stemmer.stem(q[i])
    return q