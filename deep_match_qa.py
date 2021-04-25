# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:03:50 2021

@author: Sean
"""
#%%
import os
from business import Intention
from hnsw_faiss import HNSW
from ranker import RANK
import config
import pandas as pd
#%%
it = Intention(config.train_path,
                config.ware_path,
                model_path = config.ft_path,
                kw_path= config.keyword_path
                )
hnsw = HNSW(config.w2v_path,
            config.train_path,
            config.ef_construction,
            config.M,
            config.hnsw_path
            )

#%%
import joblib
import ranker
model_path= os.path.join(config.root_path, 'model/ranking/lightgbm')
gbm = joblib.load(model_path)
#%%
query = '请问这电脑厚度是多少' 
label,score = it.predict(query)
res = pd.DataFrame()
if len(query) > 1 and '__label__1' in label:
   res = res.append(pd.DataFrame({'query': [query]*5 ,'retrieved': hnsw.search(query, 5)['custom'] , 'retr_assistance': hnsw.search(query, 5)['assistance']}))
#%%
ranked = pd.DataFrame()
#%%
ranked['question1'] = res['query']
ranked['question2'] = res['retrieved']
ranked['answer'] = res['retr_assistance']
#%%
from similarity import TextSimilarity
ts = TextSimilarity()
data = ranked
data = pd.concat([data, pd.DataFrame.from_records(data.apply(lambda row: ts.generate_all(row['question1'] , row['question2']), axis=1))], axis=1)
#%%
from matchnn import MatchingNN
matchingNN = MatchingNN()
data['matching_score'] = data.apply(lambda row: matchingNN.predict(row['question1'] , row['question2'])[1] , axis=1)
data.to_csv('result/qa_result.csv', index=False)
#%%
'''
以上代码在服务器上运行，取出qa_result.csv
'''
#%%
'''
精排结果
结合了多种相似度计算方法
lcs、edit_dist、jaccard、bm25、w2v_cos、w2v_eucl、w2v_pearson、w2v_wmd、fast_cos、fast_eucl、fast_pearson、fast_wmd、tfidf_cos、tfidf_eucl、tfidf_pearson
'''
import pandas as pd
import ranker


qa_result = pd.read_csv('result/qa_result (3).csv')
columns = [i for i in qa_result.columns if i not in ['question1' , 'question2' , 'target', 'answer']]
rank_scores = gbm.predict(qa_result[columns])
qa_result['rank_score'] = rank_scores
qa_result.to_csv('result/result.csv', index=False)

#%%

result = qa_result['rank_score'].sort_values(ascending=False)
#%%
print(qa_result['answer'].iloc[result.index[0]])


















