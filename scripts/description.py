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

class desc_ft_generator(object):
    def __init__(self, type, out_dir, save_output=True):
        self.type = type
        self.out_dir = out_dir
        self.save_output = save_output
    def IO(self):
        train_p, test_p = "downloads/train.json", "downloads/test.json"
        if self.type == "train":
            self.df=pd.read_json(train_p)
            self.df=self.df.loc[:, ["description"]].reset_index(drop=True)
        elif self.type == "full":
            train=pd.read_json(train_p)
            test=pd.read_json(test_p)
            df_full = pd.concat([train, test])
            self.df=pd.df_full.loc[:, ["description"]].reset_index(drop=True)
        else:
            print("Wrong arguments. Type should be train or full")
            assert 1 == 2
    def gen_simple_feats(self):
        def space_separator(r):
            old_pattern = r'([A-Z0-9]?[a-z0-9]+)([A-Z][a-z0-9]+)'
            new_pattern = r'\1 \2'
            return re.sub(old_pattern, new_pattern, r)
        def strip_html(r):
            old_pattern = r'<.*?>'
            new_pattern = r' '
            return re.sub(old_pattern, new_pattern, r)          
        def num_bedrooms(r):
            digits = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine" ]
            digit_map = {digits[k] : (k+1) for k in range(9)}
            digits = r'(one|two|three|four|five|six|seven|eight|nine|ten|\d+)'
            pattern = r'%s(?:-| )(?:bed|bdr|br)' % digits #Note: ?: makes it a non-capturing group
            res = re.findall(pattern, r)
            res = res[0].strip() if len(res)>0 else np.nan
            res = digit_map[res] if res in digit_map.keys() else res
            return res

        self.df["desc_clean"] = self.df["description"].apply(space_separator)
        self.df["desc_clean"] = self.df["desc_clean"].apply(strip_html)
        self.df["desc_clean"] = self.df["desc_clean"].str.replace("[''*-]", " ") # eg " *dishwasher*doorman" or "-garage -garden -fitness center"
        self.df["desc_clean"] = self.df["desc_clean"].str.lower()
        self.df["desc_clean"] = self.df["desc_clean"].str.replace(r"[\./']", " ") # eg "nice/large flat" or "a flat.it's great"
        self.df["excite_score"] = self.df["desc_clean"].apply(lambda r: \
            len(re.findall(r"!", r))/len(r)*100 if len(r)>0 else 0)
        self.df["num_bedrooms"] = self.df["desc_clean"].apply(num_bedrooms)
        self.df.loc[self.df["desc_clean"].str.contains("studio"), "num_bedrooms"] = 1.0
        self.df["num_bedrooms"] = self.df["num_bedrooms"].astype('float')
        out_path_excite = os.path.join(self.out_dir, "%s_excitement_score.csv" % self.type)
        out_path_bedrooms = os.path.join(self.out_dir, "%s_num_bedrooms.csv" % self.type)
        if self.save_output:
            self.df["excite_score"].to_csv(out_path_excite, index=False, header=False)
            self.df["num_bedrooms"].to_csv(out_path_bedrooms, index=False, header=False)
        print('cleaning and simple feats complete.')
    def gen_lsa(self):
        def prep_text(r, punc, sw, stemmer):
            """ tokenize, remove punc, remove stopwords and snowball-stem the tokens
            """
            res = [token for token in word_tokenize(r) if token not in punc]
            res = [token for token in res if token not in sw] # and len(token) > 1] # exclude one letter words
            return [stemmer.stem(token) for token in res]
        def generate_bigrams(r):
            bg_lst = [' '.join(bigram) for bigram in ngrams(r, 2)]
            return r + bg_lst
        #**************
        # STEM CORPUS *
        #**************
        sw = stopwords.words('english')     
        punc = list(string.punctuation)
        stemmer = nltk.stem.SnowballStemmer('english')
        self.df["desc_tokens"] = self.df["desc_clean"].apply(prep_text, args = (punc, sw, stemmer))
        self.df["desc_tokens"] = self.df["desc_tokens"].apply(generate_bigrams)
        print('Stemming complete.')
        #*********************
        # run DICT>TFIDF>LSA *
        #*********************
        # ******* DICT *******
        corpus = list(self.df["desc_tokens"])
        self.dictionary = corpora.Dictionary(corpus)
        self.dictionary.filter_extremes(no_below=10, no_above=.095, keep_n=3000)
        self.dictionary.compactify()
        out_path = os.path.join(self.out_dir, "%s_desc_dico.dict" % self.type)
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
        out_path = os.path.join(self.out_dir, "%s_desc_tfidf.pkl" % self.type)
        if self.save_output:
            with open(out_path, "wb") as f:
                pickle.dump(corpus_tfidf, f, pickle.HIGHEST_PROTOCOL)
        print('Tfidf generated.')
        # ******* LSA *******
        self.corpus_lsa = lsa(n_components = 30).fit_transform(X=corpus_tfidf.T)
        out_path = os.path.join(self.out_dir, "%s_desc_lsa.pkl" % self.type)
        if self.save_output:
            with open(out_path, "wb") as f:
                pickle.dump(self.corpus_lsa, f, pickle.HIGHEST_PROTOCOL)
        print('LSA generated.')

if __name__ == "__main__":
    args["type"] = sys.argv[1]
    args["out_dir"] = sys.argv[2] 
    args["save_output"] = sys.argv[3] 
    gen = desc_ft_generator(type = args["type"], out_dir = args["out_dir"], save_output=args["save_output"])
    gen.gen_simple_feats()
    gen.gen_lsa()




