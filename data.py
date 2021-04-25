#%%
import pandas as pd
import torch
from torch.utils.data import Dataset
import csv
# %%
class DataProcessForSentence(Dataset):
    def __init__(self , bert_tokenizer , file , max_char_len=103):
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs , self.seq_masks , self.seq_segments , self.labels = self.get_input(file)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self , idx):
        return self.seqs[idx] , self.seq_masks[idx] , self.seq_segments[idx] , self.labels[idx]
    
    def get_input(self, file):
        df = pd.read_csv(file,
                        delimiter="\t", 
                        encoding="utf-8"
                        )
        df['question1'] = df['question1'].apply(lambda x: "".join(x.split()))
        df['question2'] = df['question2'].apply(lambda x: "".join(x.split()))
        labels = df['target'].astype('int8').values
        tokens_seq_1 = list(map(self.bert_tokenizer.tokenize , df['question1'].values))
        tokens_seq_2 = list(map(self.bert_tokenizer.tokenize , df['question2'].values))
        
        result = list(map(self.trunate_and_pad , tokens_seq_1 , tokens_seq_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long) , torch.Tensor(seq_masks).type(torch.long) , torch.Tensor(seq_segments).type(torch.long) , torch.Tensor(labels).type(torch.long)


    def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
        if len(tokens_seq_1) > ((self.max_seq_len - 3) // 2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len -3) // 2]
        if len(tokens_seq_2) > ((self.max_seq_len - 3) // 2):
            tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len -3) // 2]
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)

        padding = [0] * (self.max_seq_len - len(seq))
        seq_mask = [1] * len(seq) + padding
        seq_segment = seq_segment + padding
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment

# %%

# %%
