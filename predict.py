# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:11:43 2021

@author: Sean
"""

#%%
import sys
import os
import torch
from config import is_cuda, root_path
from bert_model import BertConfig
from seq2seq import Seq2SeqModel
from tokenizer import load_chinese_base_vocab

#%%
class bertSeq2Seq(object):
    def __init__(self, model_path, is_cuda):
        self.word2idx = load_chinese_base_vocab()
        self.config = BertConfig(len(self.word2idx))
        self.bert_seq2seq = Seq2SeqModel(self.config)
        self.is_cuda = is_cuda
        if is_cuda:
            device = torch.device("cuda")
            self.bert_seq2seq.load_state_dict(torch.load(model_path))
            self.bert_seq2seq.to(device)
        else:
            checkpoint = torch.load(model_path,
                                    map_location=torch.device("cpu"))
            self.bert_seq2seq.load_state_dict(checkpoint)
        self.bert_seq2seq.eval()

    def generate(self, text, k=5):
        result = self.bert_seq2seq.generate(text,
                                            beam_size=k,
                                            is_cuda=self.is_cuda)
        return result
    
#%%
if __name__ == "__main__":
    bs = bertSeq2Seq(os.path.join(root_path, 'model/generative/bert.model.epoch.13'), is_cuda)
    text = '你好啊'
    print(bs.generate(text, k=1))