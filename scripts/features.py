import pandas as pd
import numpy as np
import os
import sys
import pickle
# text cleaning stuff
import re
import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
# lsa stuff
import gensim
from gensim import corpora, models
from sklearn.decomposition import TruncatedSVD as lsa

class features_generator(object):
    def __init__(self, type, out_dir, save_output=True):
        self.type = type
        self.out_dir = out_dir
        self.save_output = save_output
    def IO(self):
        train_p, test_p = "downloads/train.json", "downloads/test.json"
        if self.type == "train":
            self.df=pd.read_json(train_p)
            self.df=self.df.loc[:, ["features"]].reset_index(drop=True)
        elif self.type == "full":
            train=pd.read_json(train_p)
            test=pd.read_json(test_p)
            df_full = pd.concat([train, test])
            self.df=pd.df_full.loc[:, ["features"]].reset_index(drop=True)
        else:
            print("Wrong arguments. Type should be train or full")
            assert 1 == 2
    def clean_features(self):
        def cleaner(r):
            res = list()
            for feature in r:
                feature_clean = feature.lower()
                feature_clean = re.sub("[''*\.]", " ", feature_clean)
                res.append(feature_clean)
            return res
        self.df["features_clean"] = self.df["features"].apply(cleaner)
        out_path = os.path.join(self.out_dir, "%s_features_cleaned.csv" % self.type)
        self.df[["features_clean"]].reset_index(drop=False).to_csv(out_path, index = None)
        print('cleaning and simple feats complete.')
    def gen_lsa(self):
        def prep_features(r, punc, sw, stemmer):
            """ tokenize, remove punc, remove stopwords and snowball-stem the tokens
            """
            res = list()
            for feature in r:
                tokenized_feature = [token for token in word_tokenize(feature) if token not in punc]
                stemmed_feature = [stemmer.stem(token) for token in tokenized_feature]
                bgs=list()
                if len(stemmed_feature) > 2:
                    bgs = [' '.join(bigram) for bigram in ngrams(stemmed_feature, 2)]
                #stemmed_feature = [" ".join(stemmed_feature)]
                res += [" ".join(stemmed_feature)] + stemmed_feature + bgs
            return res

        #**************
        # STEM CORPUS *
        #**************
        sw = stopwords.words('english')     
        punc = list(string.punctuation)
        stemmer = nltk.stem.SnowballStemmer('english')
        self.df["features_new"] = self.df["features_clean"].apply(prep_features, args = (punc, sw, stemmer))
        print('Feature augmentation complete.')
        #*********************
        # run DICT>TFIDF>LSA *
        #*********************
        # ******* DICT *******
        corpus = list(self.df["features_new"])
        self.dictionary = corpora.Dictionary(corpus)
        self.dictionary.filter_extremes(no_below=5, no_above=.095, keep_n=3000)
        self.dictionary.compactify()
        out_path = os.path.join(self.out_dir, "%s_features_dico.dict" % self.type)
        if self.save_output:
            self.dictionary.save(out_path)
        corpus = [self.dictionary.doc2bow(text) for text in corpus]
        print('Dictionary generated. Its size is %s tokens' % len(self.dictionary.values()))
        # ******* TFIDF *******
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        corpus = None # free up some RAM
        corpus_tfidf = [doc for doc in corpus_tfidf]
        corpus_tfidf = gensim.matutils.corpus2csc(corpus_tfidf)
        out_path = os.path.join(self.out_dir, "%s_features_tfidf.pkl" % self.type)
        if self.save_output:
            with open(out_path, "wb") as f:
                pickle.dump(corpus_tfidf, f, pickle.HIGHEST_PROTOCOL)
        print('Tfidf generated.')
        # ******* LSA *******
        self.corpus_lsa = lsa(n_components = 30).fit_transform(X=corpus_tfidf.T)
        out_path = os.path.join(self.out_dir, "%s_features_lsa.pkl" % self.type)
        if self.save_output:
            with open(out_path, "wb") as f:
                pickle.dump(self.corpus_lsa, f, pickle.HIGHEST_PROTOCOL)
        print('LSA generated.')

if __name__ == "__main__":
    args = dict()
    args["type"] = sys.argv[1]
    args["out_dir"] = sys.argv[2] 
    args["save_output"] = sys.argv[3] 
    gen = features_generator(type = args["type"], out_dir = args["out_dir"], save_output=args["save_output"])
    gen.IO()
    gen.clean_features()
    gen.gen_lsa()





