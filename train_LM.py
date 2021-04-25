#%%
import logging
import sys
import os
from collections import defaultdict
import jieba
from gensim import corpora, models
from config import root_path
# %%
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
# %%
class Trainer(object):
    def __init__(self):
        self.data = self.data_reader(os.path.join(root_path , 'data/ranking/train.tsv')) + \
            self.data_reader(os.path.join(root_path , 'data/ranking/dev.tsv')) + \
            self.data_reader(os.path.join(root_path , 'data/ranking/test.tsv'))
        self.stopwords = open(os.path.join(root_path, 'data/stopwords.txt') , encoding='utf-8').readlines()
        self.preprocessor()
        self.train()
        self.saver()
    
    def data_reader(self , path):
        samples = []
        with open(path , 'r' , encoding='UTF-8') as f:
            for line in f:
                try:
                    q1 ,q2 ,label = line.split('\t')
                except Exception:
                    print('exception: ' , line)
                samples.append(q1)
                samples.append(q2)
        return samples
    
    def preprocessor(self):
        logging.info(" loading data.... ")
        self.data = [[
            word for word in jieba.cut(sentence) if word not in self.stopwords
        ] for sentence in self.data]
        self.freq = defaultdict(int)
        for sentence in self.data:
            for word in sentence:
                self.freq[word] += 1
        self.data = [[word for word in sentence if self.freq[word] > 1]
                     for sentence in self.data]
        logging.info(' building dictionary....')
        self.dictionary = corpora.Dictionary(self.data)
        self.dictionary.save(os.path.join(root_path, r'model/ranking/ranking.dict'))
        self.corpus = [self.dictionary.doc2bow(text) for text in self.data]
        corpora.MmCorpus.serialize(os.path.join(root_path, 'model/ranking/ranking.mm'),
                                   self.corpus)
    def train(self):
        logging.info(' train tfidf model ...')
        self.tfidf = models.TfidfModel(self.corpus, normalize=True)
        logging.info(' train word2vec model...')
        self.w2v = models.Word2Vec(min_count=2,
                                   window=2,
                                   size=300,
                                   sample=6e-5,
                                   alpha=0.03,
                                   min_alpha=0.0007,
                                   negative=15,
                                   workers=8,
                                   iter=7)
        self.w2v.build_vocab(self.data)
        self.w2v.train(self.data,
                       total_examples=self.w2v.corpus_count,
                       epochs=15,
                       report_delay=1)
        logging.info(' train fasttext model ...')
        self.fast = models.FastText(self.data,
                                    size=300,
                                    window=3,
                                    min_count=1,
                                    iter=10,
                                    min_n=3,
                                    max_n=6,
                                    word_ngrams=2)
    def saver(self):
        logging.info(' save tfidf model ...')
        self.tfidf.save(os.path.join(root_path, 'model/ranking/tfidf'))
        logging.info(' save word2vec model ...')
        self.w2v.save(os.path.join(root_path, 'model/ranking/w2v'))
        logging.info(' save fasttext model ...')
        self.fast.save(os.path.join(root_path, 'model/ranking/fast'))




# %%
if __name__ == "__main__":
    Trainer()
# %%
