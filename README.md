# FAQ-question-answer-system

## [博客链接： FAQ式问答系统](https://blog.csdn.net/qq_44193969/article/details/116128473?spm=1001.2014.3001.5502)
基于深度学习的FAQ式问答系统

所有数据集均在data文件夹里，由于数据量太大，无法上传至git

数据集百度云链接为：

链接：https://pan.baidu.com/s/1Wdi4-WKLBW4WXeGf8UnZQQ 
提取码：uivj 

整个项目结构如下：
│  bert_model.py
│  bm25.py
│  business.py
│  config.py
│  config_distil.py
│  data.py
│  data_gen.py
│  deep_match_qa.py
│  hnsw_faiss.py
│  matchnn.py
│  matchnn_utils.py
│  predict.py
│  preprocessor.py
│  ranker.py
│  seq2seq.py
│  similarity.py
│  task.py
│  test.py
│  tokenizer.py
│  tools.py
│  train.py
│  train_LM.py
│  train_matchnn.py
│  word2vec.py          
│              
├─data
│  │  chat.txt
│  │  dev.csv
│  │  stopwords.txt
│  │  test.csv
│  │  train_no_blank.csv
│  │  ware.txt
│  │  
│  ├─generative
│  │      dev.tsv
│  │      LCCC-base_test.json
│  │      LCCC-base_train.json
│  │      LCCC-base_valid.json
│  │      test.tsv
│  │      train.tsv
│  │      
│  ├─intention
│  │      business.test
│  │      business.train
│  │      key_word.txt
│  │      
│  └─ranking
│          dev.tsv
│          test.tsv
│          train.tsv
│          
├─lib
│  └─bert
│          config.json
│          pytorch_model.bin
│          vocab.txt
│          
├─log
│      
│      
├─model
│  ├─generative 
│  │      
│  ├─intention
│  │        
│  ├─ranking
│  │      
│  └─retrieval
│          
├─result
│              



python deep_match_qa.py执行



项目结构在线看如果比较乱的话，可以把代码拉到本地用typora查看
