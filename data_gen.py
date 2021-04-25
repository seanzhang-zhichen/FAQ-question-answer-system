#%%
import sys
import os
import pandas as pd
from config import root_path
import json
# %%
with open(os.path.join(root_path , 'data/generative/train.tsv') , 'w' , encoding='utf-8') as train:
    count = 0
    with open(os.path.join(root_path, 'data/generative/LCCC-base_train.json') , 'r' , encoding='utf-8') as r :
        js_file = json.load(r)
        for item in js_file:
            if count >= 100000:
                break
            if len(item) == 2:
                line = '\t'.join(item)
                train.write(line)
                train.write('\n')
                count += 1

#%%

with open(os.path.join(root_path, 'data/generative/dev.tsv'), 'w' , encoding= 'utf-8') as dev:
    count = 0
    with open(os.path.join(root_path, 'data/generative/LCCC-base_valid.json'), 'r',  encoding='utf-8') as r:
        js_file = json.load(r)
        for item in js_file:
            
            if len(item) == 2:
                line = '\t'.join(item)
                dev.write(line)
                dev.write('\n')
                count += 1
                
#%%
with open(os.path.join(root_path, 'data/generative/test.tsv'), 'w' , encoding='utf-8') as test:
    count = 0
    with open(os.path.join(root_path, 'data/generative/LCCC-base_test.json'), 'r' , encoding='utf-8') as r:
        js_file = json.load(r)
        for item in js_file:
            
            if len(item) == 2:
                line = '\t'.join(item)
                test.write(line)
                test.write('\n')
                count += 1
# %%
