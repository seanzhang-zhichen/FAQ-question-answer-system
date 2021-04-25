#%%
import math
import sys
from collections import Counter
import os
import csv

# %%
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
import joblib
from config import root_path
# %%
class BM25(object):
    def __init__(self, do_train=True , save_path=os.path.join(root_path, 'model/ranking/')):
        if do_train:
            self.data = pd.read_csv(os.path.join(root_path , 'data/ranking/train.tsv'), sep='\t', header=None,
                                    quoting=csv.QUOTE_NONE, names=['question1', 'question2', 'target'])
            self.idf, self.avgdl = self.get_idf()
            self.saver(save_path)
        else:
            self.stopwords = self.load_stop_word()
            self.load(save_path)
    

    def load_stop_word(self):
        stop_words = os.path.join(root_path, 'data/stopwords.txt')
        stopwords = open(stop_words , 'r' , encoding='utf-8').readlines()
        stopwords = [w.strip() for w in stop_words]
        return stopwords
    
    def tf(self , word, count):
        return count[word] / sum(count.values())

    def n_containing(self , word , count_list):
        return sum(1 for count in count_list if word in count)

    def cal_idf(self , word , count_list):
        return math.log(len(count_list)) / (1 + self.n_containing(word , count_list))

    def get_idf(self):
        self.data['question2'] = self.data['question2'].apply(lambda x: " ".join(jieba.cut(x)))
        idf = Counter([y for x in self.data['question2'].tolist() for y in x.split()])
        idf = {k: self.cal_idf(k, self.data['question2'].tolist()) for k, v in idf.items()}
        avgdl = np.array([len(x.split()) for x in self.data['question2'].tolist()]).mean()
        return idf, avgdl
    
    def saver(self , save_path):
        joblib.dump(self.idf , save_path + 'bm25_idf.bin')
        joblib.dump(self.avgdl , save_path + 'bm25_avgdl.bin')
    
    def load(self , save_path):
        self.idf = joblib.load(save_path + 'bm25_idf.bin')
        self.avgdl = joblib.load(save_path + 'bm25_avgdl.bin')

    def bm_25(self , q , d , k1=1.2 , k2=200 , b=0.75):
        stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
        words = pseg.cut(q)  # 切分查询式
        fi = {}
        qfi = {}
        for word, flag in words:
            if flag not in stop_flag and word not in self.stopwords:
                fi[word] = d.count(word)
                qfi[word] = q.count(word)
        K = k1 * (1 - b + b * (len(d) / self.avgdl))  # 计算K值
        ri = {}
        for key in fi:
            ri[key] = fi[key] * (k1+1) * qfi[key] * (k2+1) / ((fi[key] + K) * (qfi[key] + k2))  # 计算R

        score = 0
        for key in ri:
            score += self.idf.get(key, 20.0) * ri[key]
        return score
#%%
if __name__ == '__main__':
    bm25 = BM25(do_train = True)
    
# %%
