# FAQ-question-answer-system

基于深度学习的FAQ式问答系统

所有数据集均在data文件夹里，由于数据量太大，无法上传至git

数据集百度云链接为：



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