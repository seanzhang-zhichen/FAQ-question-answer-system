#%%
import torch
import os
#%%
root_path = os.path.abspath(os.path.dirname(r"D:/WorkPlace/Chatbot/DS/linux_code/"))

train_raw = os.path.join(root_path, 'data/chat.txt')
dev_raw = os.path.join(root_path, 'data/开发集.txt')
test_raw = os.path.join(root_path, 'data/测试集.txt')
ware_path = os.path.join(root_path, 'data/ware.txt')
# %%
sep = '[SEP]'
# %%
train_path = os.path.join(root_path, 'data/train_no_blank.csv')
dev_path = os.path.join(root_path, 'data/dev.csv')
test_path = os.path.join(root_path, 'data/test.csv')
# %%
# intention
business_train = os.path.join(root_path, 'data/intention/business.train')
business_test = os.path.join(root_path, 'data/intention/business.test')
keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')
# %%
# fasttext
ft_path = os.path.join(root_path, "model/intention/fastext")

''' Retrival '''
# Embedding
w2v_path = os.path.join(root_path, "model/retrieval/word2vec")
# %%
# HNSW parameters
ef_construction = 3000  # ef_construction defines a construction time/accuracy trade-off
M = 64  # M defines tha maximum number of outgoing connections in the graph
hnsw_path = os.path.join(root_path, 'model/retrieval/hnsw.bin')
#%%
#matchnn
max_sequence_length = 103


#%%
#Bert-generate
base_chinese_bert_vocab = os.path.join(root_path, 'lib/bert/vocab.txt')


max_length = 103

batch_size = 32

lr = 0.001

bert_chinese_model_path = os.path.join(root_path, 'lib/bert/pytorch_model.bin')


log_path = os.path.join(root_path, 'log/distil.log')

max_grad_norm = 5.0

gradient_accumulation = 2.0
# %%
is_cuda = True
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %%
'''
# %%
from numpy import*


def load_dataset(filename):
    # 将数据存储到numpy的array中
    fr = open(filename, 'r' , encoding="UTF-8-sig")
    data_mat = []
    for line in fr.readlines():
        line_arr = []
        curline = line.strip().split('\t')
        for i in range(len(curline)):
            line_arr.append(curline[i])
        data_mat.append(line_arr)
    data_arr = array(data_mat)


if __name__ == '__main__':
    filename = 'C:/Users/Admin/Desktop/DS/code/data/ranking/ranking_datasets/ranking_datasets/task3_train.txt'
    load_dataset(filename)
# %%
import csv
filename = 'C:/Users/Admin/Desktop/DS/code/data/ranking/ranking_datasets/ranking_datasets/task3_train.txt'
lines = csv.reader(open(filename, "rt", encoding="UTF-8-sig"))
dataset = list(lines)
for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
# %%
from numpy import*
filename = 'C:/Users/Admin/Desktop/DS/code/data/ranking/ranking_datasets/ranking_datasets/task3_train.txt'
fr = open(filename, 'r' , encoding="UTF-8-sig")
data_mat = []
for line in fr.readlines():
    line_arr = []
    curline = line.strip().split('\t')
    for i in range(len(curline)):
        line_arr.append(curline[i])
    data_mat.append(line_arr)
data_arr = array(data_mat)
# %%

# %%
with open('C:/Users/Admin/Desktop/DS/code/data/ranking/ranking_datasets/ranking_datasets/data.tsv','w+',encoding="utf8")as t:
    with open("C:/Users/Admin/Desktop/DS/code/data/ranking/ranking_datasets/ranking_datasets/task3_train.txt",'r',encoding='UTF-8-sig')as f:
        # print(f.readlines())
        


        for line in f.readlines():
            # print(line)
            line_list = line.strip().split('\t')    #去掉str左右端的空格并以空格分割成list
            # print(line_list)
            hbaseRowID_list = line_list[0:3]    #取前三个list中的元素
            # print(hbaseRowID_list)
            # hbaseRowID = line_list[0]+line_list[1]+line_list[2]
            hbaseRowID = " ".join(hbaseRowID_list) #连接list
            # print(hbaseRowID)
            # print(type(line_list))
            # print(line_list)
            line_list[2] = hbaseRowID
            tsv_list = line_list[2:]
            tsv_list = '\t'.join(tsv_list)
            print(tsv_list)
            t.write(tsv_list+'\n')

# %%
with open('C:/Users/Admin/Desktop/DS/code/data/ranking/ranking_datasets/ranking_datasets/data2.tsv','w+',encoding="utf8")as t:
    with open("C:/Users/Admin/Desktop/DS/code/data/ranking/ranking_datasets/ranking_datasets/task3_train.txt",'r',encoding='UTF-8-sig')as f:
        # print(f.readlines())
        for line in f.readlines():
            # print(line)
            line_list = line.strip().split('\t')    #去掉str左右端的空格并以空格分割成list
            # print(line_list)
            hbaseRowID_list = line_list[0:3]    #取前三个list中的元素
            # print(hbaseRowID_list)
            # hbaseRowID = line_list[0]+line_list[1]+line_list[2]
            hbaseRowID = "\t".join(hbaseRowID_list) #连接list
            # print(hbaseRowID)
            # print(type(line_list))
            # print(line_list)
            line_list[2] = hbaseRowID
            tsv_list = line_list[2:]
            tsv_list = '\t'.join(tsv_list)
            #print(tsv_list)
            t.write(tsv_list+'\n')

# %%
'''
'''
#%%
import pandas as pd
data1 = pd.read_csv('C:/Users/Admin/Desktop/DS/code/data/ranking/ranking_datasets/ranking_datasets/atec_nlp_sim_train.csv' , delimiter="\t", header=None , names=['index','question1', 'question2', 'target'])
# %%
data1.drop(['index'] ,axis=1 , inplace=True)
# %%
data2 = pd.read_csv('C:/Users/Admin/Desktop/DS/code/data/ranking/ranking_datasets/ranking_datasets/atec_nlp_sim_train_add.csv' , delimiter="\t", header=None , names=['index','question1', 'question2', 'target'])
# %%
data2.drop(['index'] ,axis=1 , inplace=True)
# %%
# %%
data3 = pd.read_csv('C:/Users/Admin/Desktop/DS/code/data/ranking/train2.tsv' , delimiter="\t"  , encoding='gbk' , header=None , names=['question1', 'question2', 'target'])
# %%
data3
#%%
data1.to_csv('C:/Users/Admin/Desktop/DS/code/data/ranking/dev.tsv' , index=False , sep='\t' , encoding ='utf-8')
data2.to_csv('C:/Users/Admin/Desktop/DS/code/data/ranking/test.tsv' , index=False , sep='\t' , encoding ='utf-8')

# %%
data_all = pd.concat([data1,data2 ,data3] , axis=0)
# %%

data_all
#%%
data_all:pd.DataFrame = data_all.sample(frac=1.0) 


# %%
from sklearn.utils import shuffle  # 用于数据的随机排列，也可不用
# %%

# %%
data_all.reset_index(inplace=True)
data_all.drop(['level_0'] ,axis=1 , inplace=True)
# %%
'''
