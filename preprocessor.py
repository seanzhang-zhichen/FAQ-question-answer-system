#%%
import logging
import re
import sys
import os
import pathlib
# %%
import numpy as np
import pandas as pd
from config import dev_raw, root_path, sep, test_raw, train_raw
import config
# %%
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)
# %%
def filter_content(sentence):

    sentence = re.sub(
        r"#E\-[\w]*(抱拳|傲慢|得意|蛋糕|呕吐|闭嘴|礼物|yaoping|柠檬|流泪|怒火|撇嘴|太阳|咒骂|糗|猪猪|足球|磕头|大兵|电话|灯泡|飞鸟|奋斗|高兴|击打|饥饿|咖啡|口罩|骷髅|可乐|疯狂|白眼|阴险|叹气|奸笑|发呆|害羞|飞吻|怒火|悲伤|胜利|生病|弱|可怜|咖啡|酷酷|眩晕|流泪|发抖|难过|右哼哼|惊恐|悲伤|犯困|愤怒|凋谢|哈欠|拥抱|抓狂|鄙视|时间|啤酒|勾引|左哼哼|月亮|偷笑|震惊|惊讶|跳跳|瞌睡|可爱|衰样|好|憨笑|水果|色色|黑线|微笑|流汗|握手|心碎|问号|大哭|亲亲|抠鼻|拜拜|鬼脸|香吻|米饭|花朵|尴尬|擦汗|安慰|委屈|调皮|爱心|我一定尽力为您解答的哦|很棒|鼓掌)+",
        "α", str(sentence))
    sentence = re.sub(r"#E\-[\w]+\[数字x]", "α", sentence)
    sentence = re.sub(r"\[ORDERID_[\d]+]", "[订单x]", sentence)
    sentence = re.sub(r"\[数字x]", "γ", sentence)
    sentence = re.sub(r"\[链接x]", "ε", sentence)
    sentence = re.sub(r"\[表情]", "α", sentence)
    sentence = re.sub("<sep>", sep, sentence)
    sentence = re.sub("<SEP>", sep, sentence)
    sentence = re.sub(
        r"(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?",
        "ε", sentence)
    sentence = re.sub(r"(http|ftp|https):\/\/ε", "ε", sentence)
    sentence = re.sub(r"[\d]+.*[\d]+", "γ", sentence)
    sentence = re.sub(r"【收到不支持的消息类型，暂无法显示】", " ", sentence)

    sentence = re.sub(r"#E\-[s]*(ν|γ|π|ζ|ρ|α|ε)*", "α", sentence)
    sentence = re.sub("α", " ", sentence)
    sentence = re.sub("ε", "[链接x]", sentence)
    sentence = re.sub("γ", "[数字x]", sentence)

    return sentence
# %%
def read_file(path, is_train=False):

    chat = []

    with open(path, 'r' , encoding='utf-8') as f:

        tmp = []
        sessions = set()
        session_id, custom_id, is_assistance, content = '', '', '', []
        for lines in f:
            if len(sessions) > 300000:  
                break
            line = lines.strip().replace(' ', '').split('\t')
            if len(line) < 5: 
                continue
            if is_train:
                session_id_in_doc, custom_id_in_doc, is_assistance_in_doc = \
                    line[0], line[1], line[2]
            else:
                session_id_in_doc, custom_id_in_doc, is_assistance_in_doc = \
                    line[2], line[1], line[3]
            sessions.add(session_id_in_doc)
            if session_id != session_id_in_doc and session_id != '':
                fc = filter_content(content)
                if fc != '':
                    tmp.append([
                        session_id, 'custom'
                        if str(is_assistance) == '0' else 'assistance', fc
                    ])
                    content = []
                chat.extend(tmp)
                tmp = []
                session_id, custom_id = session_id_in_doc, custom_id_in_doc
            else:
                if is_assistance != is_assistance_in_doc and \
                        is_assistance != '':
                    content = filter_content(content)
                    is_assistance = 'custom' if str(
                        is_assistance) == '0' else 'assistance'
                    if content != '':
                        tmp.append([session_id, is_assistance, content])
                    is_assistance = is_assistance_in_doc
                    content = [line[-1]]
                else:
                    content.append(line[-1])
                    is_assistance = is_assistance_in_doc
                    session_id, _ = session_id_in_doc, custom_id_in_doc
    if content != '':
        tmp.append([
            session_id,
            'custom' if str(is_assistance) == '0' else 'assistance',
            filter_content(content)
        ])
    chat.extend(tmp)
    return chat

# %%
def clean(sent, sep='<'):
    sent = re.sub(r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+",
                  "", sent)
    i = 0
    tmp = []
    while i < len(sent):
        if sent[i] != sep:
            tmp.append(sent[i])
            i += 1
        else:
            tmp.append(sent[i:i + 5])
            i += 5
    # 过滤短文本？
    return " ".join(tmp)
# %%
def generate_data(filepath, save=True, to_file=None, pair=False):
    data = read_file(filepath, 'train' in to_file)
    data = pd.DataFrame(data, columns=['session_id', 'role', 'content'])
    if 'train' in to_file:
        data = data[(data['content'].str.len() <= 128)
                    & (data['content'].str.len() > 1)].reset_index(drop=True)
    data = data.reset_index()
    data['index'] = data['index'].apply(lambda x: x - 1
                                        if x % 2 == 1 else x)
    data = data.pivot_table(index=['index', 'session_id'],
                            columns='role',
                            values='content',
                            aggfunc='first').reset_index()
    data = data[['session_id', 'custom',
                'assistance']].dropna().reset_index(drop=True)

    if save:
        data.to_csv('{}.csv'.format(to_file), index=False)
    return data
# %%
if __name__ == "__main__":
    dev = generate_data(dev_raw,
                        save=True,
                        to_file=os.path.join(root_path, 'data/dev'),
                        pair=True)
    logging.info('Dev set created.')
    

# %%
    test = generate_data(test_raw,
                         save=True,
                         to_file=os.path.join(root_path, 'data/test'),
                         pair=True)
    logging.info('test set created.')
# %%
    data = generate_data(train_raw,
                         save=True,
                         to_file=os.path.join(
                             root_path, 'data/train_no_blank'),
                         pair=True)
    logging.info('training set created.')
# %%

# %%

# %%

# %%
