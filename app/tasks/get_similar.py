from huey import FileHuey
import pathlib
path = str(pathlib.Path(__file__).parent.resolve()) + "/%s" % "storage"
print(path)
huey = FileHuey(path=path)

import nltk
nltk.download('omw-1.4')


from ..text_summarization import text_summarization
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import gensim.downloader as api
import time
from hashlib import sha256

def similar_cache_by_request_hash(request_hash):
    result_matrix = huey.get(request_hash, peek=True)
    request_corpus = huey.get(f"{request_hash}_corpus", peek=True)
    if result_matrix:
        print("Found: %s" % result_matrix)
        return result_matrix, request_corpus
    else:
        return None,None


def similar_cache(doc_list):
    if isinstance(doc_list, list):
        flat_list = [item for sublist in doc_list for item in sublist]
        doc_list_str = ("".join(flat_list)).encode("utf-8")
    else:
        doc_list_str = doc_list.encode("utf-8")

    doc_list_to_bytes = bytes(doc_list_str)
    request_hash = sha256(doc_list_to_bytes).hexdigest()
    print("Request Hash: %s" % request_hash)
    storage = huey.get(request_hash, peek=True)
    if storage:
        print("Found: %s" % storage)
        return storage, request_hash

    return None, request_hash


@huey.task(context=True)
def find_similar(corpus, task=None):
    task_id = task.id
    print("~~ TaskID: %s" % task_id)

    if len(corpus) == 0:
        print("Empty corpus, returning.")
        return

    c, request_hash = similar_cache(corpus)
    if c:
        return c

    print("Preprocessing...")
    t = text_summarization()
    doc_list = []
    for line in corpus.split("\n"):
        tokens = t.prepare_text_for_lda(line)
        doc_list.append(tokens)
    print("Done with preprocessing.")


    print("Getting word2vec model")
    start_time = time.time()
    # https://github.com/RaRe-Technologies/gensim-data
    #print(api.load("word2vec-google-news-300", return_path=True))
    #/Users/gvannoni/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz
    model = api.load('word2vec-google-news-300')
    exec_time = time.time()-start_time
    print(f"{exec_time}")

    print("----------")
    print(doc_list)
    print("----------")

    dictionary = Dictionary(doc_list)

    doc2bow_list = []
    for item in doc_list:
        print(f"Converting {item} into bag of words...")
        doc2bow_list.append(dictionary.doc2bow(item))

    print("Creating TFIDF model")
    tfidf = TfidfModel(doc2bow_list)

    print("Creating Similarity Index")
    termsim_index = WordEmbeddingSimilarityIndex(model)

    print("Creating Sparse Term Similarity Matrix")
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)

    vectorized_dict = []
    counter = 0
    print("Vectorizing using TFIDF...")
    for item in doc2bow_list:
        sentence = tfidf[item]
        vectorized_dict.append(sentence)
    print("Done with vectorization.")

    print("~~ Vectorized ~~")
    print(vectorized_dict)
    print("~~ Vectorized ~~")


    w, h = len(vectorized_dict), len(vectorized_dict)
    results = [[0 for x in range(w)] for y in range(h)] # Initialize everything as 0

    for i in range(0,len(vectorized_dict)):
        myitem = doc_list[i]
        print(f"Processing matches for {myitem}...")
        for j in range(0,len(vectorized_dict)):
            if i != j:
                match_item = doc_list[j]
                similarity_score = termsim_matrix.inner_product(vectorized_dict[i], vectorized_dict[j], normalized=(True, True))
                print(f" {i}x{j}: {similarity_score:.3f} for {match_item}")
                results[i][j] = similarity_score

    print("Match result 2D Matrix:")
    print(results)
    huey.put(request_hash, results)
    huey.put(f"{request_hash}_corpus", corpus)
    return results
