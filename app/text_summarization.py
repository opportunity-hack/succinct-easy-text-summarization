from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import sys
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import itertools
from collections import OrderedDict
import os
from sklearn.feature_extraction.text import CountVectorizer
from itertools import islice
import seaborn as sns
import base64
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import wordnet as wn
import gensim
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import time
import spacy
import pytextrank

import matplotlib
matplotlib.use('Agg')




# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")
# add PyTextRank to the spaCy pipeline
# https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
nlp.add_pipe("textrank") #not using positionrank because position likely doesn't matter in this context for our app


print("Downloading wordnet")
start_time = time.time()
nltk.download('wordnet')
exec_time = time.time()-start_time
print(f"{exec_time}")

print("Downloading stopwords")
start_time = time.time()
nltk.download('stopwords')
exec_time = time.time()-start_time
print(f"{exec_time}")


class text_summarization:

    num_ngrams = ""
    matrix_sparcity = ""
    term_examples = OrderedDict()
    tf_idf_results = ""
    term_plot = ""
    tfidf_plot = ""

    # Inputs
    ngram_start = int()
    ngram_end = int()
    min_df = float()
    max_df = float()


    en_stop = set(nltk.corpus.stopwords.words('english'))

    def smart_truncate(self, content, length=140, suffix='...'):
        if len(content) <= length:
            return content
        else:
            return ' '.join(content[:length+1].split(' ')[0:-1]) + suffix

    def clean_spaces(self, strval):
        strval = strval.replace("\n", " ").replace("\t", " ")
        strval = re.sub("\s\s+", " ", strval)
        return strval


    def tokenize(self, cleaned_string):
        import spacy
        #spacy.load('en')
        from spacy.lang.en import English
        parser = English()
        lda_tokens = []
        tokens = parser(cleaned_string)

        for token in tokens:
            if token.orth_.isspace():
                continue
            if token.orth_.isnumeric():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lowercase = token.lower_
                lowercase = lowercase.replace("'","")
                lda_tokens.append(lowercase)
        return lda_tokens


    def get_lemma(self, word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    from nltk.stem.wordnet import WordNetLemmatizer
    def get_lemma2(self, word):
        return WordNetLemmatizer().lemmatize(word)

    def prepare_text_for_lda(self, text):
        tokens = self.tokenize(text)
        tokens = [token for token in tokens if len(token) > 1]
        tokens = [token for token in tokens if token not in self.en_stop]
        tokens = [self.get_lemma(token) for token in tokens]
        return tokens


    def get_common_words(self, word_nested_list):

        flat_list = [item[0] for sublist in word_nested_list for item in sublist]
        print("Flat",flat_list)
        long_string = " ".join(flat_list)
        all_words = long_string.split(" ")
        all_words = [token for token in all_words if token not in self.en_stop]
        all_words = [token for token in all_words if len(token) > 1]
        all_words_lowercase = (map(lambda x: x.lower(), all_words))
        word_counter = Counter(all_words_lowercase)
        most_common_words = word_counter.most_common(20)
        return most_common_words




    def get_tags(self, text):
        doc = nlp(text)

        tags = []
        for phrase in doc._.phrases[:20]:
            print("Text:",phrase.text)
            print(" Rank",phrase.rank, "Count:",phrase.count)
            print(" Chunks",phrase.chunks)
            if phrase.rank >= 0.05:
                tags.append( (phrase.text,phrase.rank) )
        return tags

    # https://www.tutorialspoint.com/gensim/gensim_creating_lsi_and_hdp_topic_model.htm
    # https://colab.research.google.com/github/explosion/spacy-notebooks/blob/master/notebooks/conference_notebooks/pydays/topic_modelling.ipynb#scrollTo=g-EyYC0KJ-Nx
    def get_topics(self, text_data):
        print(text_data)
        dictionary = corpora.Dictionary(text_data)
        corpus = [dictionary.doc2bow(text) for text in text_data]

        # HDP, the Hierarchical Drichlet Process is an unsupervised topic model which figures out the number of topics on it's own.
        hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)

        topics = hdpmodel.show_topics(num_words=5, log=False, formatted=False)
        #hdptopics = [[word for word, prob in topic] for topicid, topic in hdpmodel.show_topics(formatted=False)]
        return topics

    def normalize_text(self, pd_series):
        pd_series.replace('\n',' ',inplace=True,regex=True)
        pd_series.replace('((http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?)','',inplace=True,regex=True)
        pd_series.replace('[!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]',' ',inplace=True,regex=True)
        pd_series.dropna(axis=0, how='all', inplace=True)
        return pd_series

    def stem_text(self, pd_series):
        pd_series['TOKENIZED']=pd_series.apply(lambda x : x.split(" "))

        stemmer = SnowballStemmer("english")
        result = pd.DataFrame()

        # Stem array of words
        result['stemmed'] = pd_series['TOKENIZED'].apply(lambda x: [stemmer.stem(y) for y in x])

        # Connect array of words back together
        result['stemmed'] = result['stemmed'].apply(lambda x: " ".join(x))
        return result

    def random_string(self, length):
        import string, random
        return ''.join(random.sample(string.ascii_lowercase, length))

    def gen_plot(self, title, x_title, y_title, data):
        g = sns.catplot(x=x_title, y=y_title, data=data, color="b", kind="bar", aspect=2.5)
        g.set_xticklabels(rotation=75)
        g.fig.suptitle(title)

        # We're doing this to make sure that two sessions don't stomp on file generation
        pfile = "/tmp/figure_%s.png" % self.random_string(10)
        #fig, ax = plt.subplots(figsize=(10, 10)) # set size
        #ax.margins(0.2) # Optional, just adds 5% padding to the autoscaling))
        #sns.set_style("dark")

        plt.savefig(pfile, format="png", bbox_inches="tight")
        with open(pfile, "rb") as afile:
            b64_value = base64.b64encode(afile.read()).decode("utf-8")
        os.remove(pfile) # Once the base64 image is made, we don't need the file

        return b64_value


    def get_text_stats(self, pandas_col_name, csv_data):
        print("Processing Column: |" + pandas_col_name + "|")
        orig_data = pd.Series(csv_data[str(pandas_col_name)].values.ravel())

        data = self.normalize_text(orig_data)
        print(data)
        # Print out specific rows that match a string - useful for debugging
        # print(data[data.str.contains("money", case=False)])

        df = self.stem_text(data)


        # Initialize the vectorizer with new settings and check the new vocabulary length
        # n-gram: https://en.wikipedia.org/wiki/N-gram
        cvec = CountVectorizer(stop_words='english', min_df=self.min_df, max_df=self.max_df, ngram_range=(self.ngram_start,self.ngram_end))

        cvec.fit(df.stemmed)
        # Print 20 items
        # print(list(islice(cvec.vocabulary_.items(), 20)))

        # Check how many total n-grams we have
        self.num_ngrams = len(cvec.vocabulary_)

        # Our next move is to transform the document into a “bag of words” representation which essentially is just a separate column
        #  for each term containing the count within each document. After that, we’ll take a look at the sparsity of this representation
        #  which lets us know how many nonzero values there are in the dataset.
        #  The more sparse the data is the more challenging it will be to model, but that’s a discussion for another day.
        cvec_counts = cvec.transform(df.stemmed)
        print('sparse matrix shape:', cvec_counts.shape)
        print('nonzero count:', cvec_counts.nnz)
        self.sparsity = (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1]))
        print('sparsity: %.2f%%' % self.sparsity)

        # Let’s look at the top 20 most common terms
        occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
        counts_df = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
        self.common_terms = counts_df.sort_values(by='occurrences', ascending=False)[["occurrences","term"]]
        print(self.common_terms)

        # Use this so that text is displayed in the same sorted order
        self.term_examples = OrderedDict()

        # Term Examples
        for term in counts_df.sort_values(by='occurrences', ascending=False).head(100)["term"]:
            print("-> ", term)
            indexesOfMatchList = df.index[df["stemmed"].str.contains(term)].tolist()
            indexesOfMatch = set()
            for ioml in indexesOfMatchList:
                indexesOfMatch.add(ioml)

            # Only print top 5 examples
            for i, val in enumerate(itertools.islice(indexesOfMatch, 5)):
                val_concat = self.smart_truncate(df["stemmed"][val]) + "-> " + self.smart_truncate(orig_data[val])
                if term in self.term_examples:
                    self.term_examples[term].add(val_concat)
                else:
                    self.term_examples[term] = set()
                    self.term_examples[term].add(val_concat)

                print("    |-> Original Text: ", self.smart_truncate(orig_data[val]))
                print("    |-> Stemmed: ", self.smart_truncate(df["stemmed"][val]))
            #print(df.loc[df["stemmed"].str.contains(term)].head(3))



        # ----------[ TF-IDF Results ]----------------------
        #
        #
        # And that about wraps it up for Tf-idf calculation. As an example, you can jump straight to the end using the TfidfVectorizer class:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tvec = TfidfVectorizer(min_df=self.min_df, max_df=self.max_df, stop_words='english', ngram_range=(self.ngram_start,self.ngram_end))
        tvec_weights = tvec.fit_transform(df.stemmed)
        weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
        weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
        self.tf_idf_results = weights_df.sort_values(by='weight', ascending=False)
        print(self.tf_idf_results.head(20))

        self.term_plot = self.gen_plot("Common Terms", "term", "occurrences", self.common_terms.head(30))
        self.tfidf_plot = self.gen_plot("TFIDF Weightings", "term", "weight", self.tf_idf_results.head(30))
