#%%
import logging
import sys
import os
import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import config
from preprocessor import clean
import hnswlib
# %%
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)
# %%
def wam(sentence, w2v_model):
    arr = []
    for s in clean(sentence).split():
        if s not in w2v_model.wv.vocab.keys():
            arr.append(np.random.randn(1, 300))
        else:
            arr.append(w2v_model.wv.get_vector(s))
    return np.mean(np.array(arr), axis=0).reshape(1, -1)

#%%
class HNSW(object):
    def __init__(self,
                 w2v_path,
                 data_path=None,
                 ef=config.ef_construction,
                 M=config.M,
                 model_path=config.hnsw_path):
        self.w2v_model = KeyedVectors.load(w2v_path)

        self.data = self.data_load(data_path)
        if model_path and os.path.exists(model_path):
            # 加载
            self.hnsw = self.load_hnsw(model_path)
        else:
            # 训练
            self.hnsw = \
                self.build_hnsw(os.path.join(config.root_path, 'model/retrieval/hnsw.bin'),
                                ef=ef,
                                m=M)

    def data_load(self, data_path):
        
        data = pd.read_csv(
            data_path)
        data['custom_vec'] = data['custom'].apply(
            lambda x: wam(x, self.w2v_model))
        data['custom_vec'] = data['custom_vec'].apply(
            lambda x: x[0][0] if x.shape[1] != 300 else x)
        data = data.dropna()
        return data

    def build_hnsw(self, to_file, ef=2000, m=64):
      
        logging.info('build_hnsw')
        dim = self.w2v_model.vector_size
        num_elements = self.data['custom'].shape[0]
        hnsw = np.stack(self.data['custom_vec'].values).reshape(-1, 300)

       
        p = hnswlib.Index(space='l2',
                          dim=dim)  # possible options are l2, cosine or ip
        p.init_index(max_elements=num_elements, ef_construction=ef, M=m)
        p.set_ef(10)
        p.set_num_threads(8)
        p.add_items(hnsw)
        logging.info('Start')
        labels, distances = p.knn_query(hnsw, k=1)
        print('labels: ', labels)
        print('distances: ', distances)
        logging.info("Recall:{}".format(
            np.mean(labels.reshape(-1) == np.arange(len(hnsw)))))
        p.save_index(to_file)
        return p

    def load_hnsw(self, model_path):
        
        hnsw = hnswlib.Index(space='l2', dim=self.w2v_model.vector_size)
        hnsw.load_index(model_path)
        return hnsw

    def search(self, text, k=5):
      
        test_vec = wam(clean(text), self.w2v_model)
        q_labels, q_distances = self.hnsw.knn_query(test_vec, k=k)
        return pd.concat(
            (self.data.iloc[q_labels[0]]['custom'].reset_index(),
             self.data.iloc[q_labels[0]]['assistance'].reset_index(drop=True),
             pd.DataFrame(q_distances.reshape(-1, 1), columns=['q_distance'])),
            axis=1)

#%%
if __name__ == "__main__":
    hnsw = HNSW(config.w2v_path,
                config.train_path,
                config.ef_construction,
                config.M,
                config.hnsw_path
                )
#%%
    test = '在手机上下载'
    result = hnsw.search(test, k=10)
#%%

# %%
'''
# %%
data_path = config.train_path
w2v_path = config.w2v_path
w2v_model = KeyedVectors.load(w2v_path)
def load_data(data_path):
    '''
   # @description: 读取数据，并生成句向量
   # @param {type}
   # data_path：问答pair数据所在路径
   # @return: 包含句向量的dataframe
'''
    data = pd.read_csv(
        data_path)
    data['custom_vec'] = data['custom'].apply(
        lambda x: wam(x, w2v_model))
    data['custom_vec'] = data['custom_vec'].apply(
        lambda x: x[0][0] if x.shape[1] != 300 else x)
    data = data.dropna()
    return data
'''
# %%
'''
logging.info('Building hnsw index.')
data = load_data(data_path)

vecs = np.stack(data['custom_vec'].values).reshape(-1, 300)
vecs = vecs.astype('float32')
# %%
dim = w2v_model.vector_size

num_elements = vecs.shape[0]
index = hnswlib.Index(space='cosine' , dim=dim)
#%%
index.init_index(max_elements = num_elements , ef_construction=config.ef_construction , M=config.M)
#%%
index.set_num_threads(8)
print('xb: ', vecs.shape)
print('dtype: ', vecs.dtype)
#%%
index.get_current_count()
#%%
index.set_ef(50) # ef should always be > k
index.add_items(vecs)
#%%
index.save_index(config.hnsw_path)
#%%
self.evaluate(vecs[:10000])
index.save_index(to_file)
'''
