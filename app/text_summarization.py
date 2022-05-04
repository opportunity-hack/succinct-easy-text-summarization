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

    def smart_truncate(self, content, length=140, suffix='...'):
        if len(content) <= length:
            return content
        else:
            return ' '.join(content[:length+1].split(' ')[0:-1]) + suffix

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
