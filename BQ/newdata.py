from hanziconv import HanziConv
from pypinyin import lazy_pinyin
from cnradical import Radical, RunOption
import os
import re
import jieba
import pickle
from zhon.hanzi import punctuation as c_pun
import string

e_punctuation = string.punctuation
print(e_punctuation)
radical = Radical(RunOption.Radical)
def remove_upprintable_chars(s):
    """移除所有不可见字符"""
    return ''.join(x for x in s if x.isprintable())

def text_clean(data):
    new_sent = []
    for line in data:
        line = line.lower()
        line = line.replace(' ','')
        line = HanziConv.toSimplified(line)
        line = remove_upprintable_chars(line)
        new_sent.append(line)
    return new_sent



def split_char(text):
    new_text = ''
    for i in text:
        new_text += i + ' '
    return new_text.strip()

def pinyin_char(text):
    pattern = re.compile(u'[^\u4e00-\u9fa5]')
    x = pattern.findall(text)
    str = ''
    for i in text:
        if i in x:
            str += i + ' '
        else:
            str += i
    py = lazy_pinyin(str)
    res = []
    for i in py:
        res.extend(i.split())
    pinyin = ' '.join(res)
    return pinyin.strip()

def get_radical(text):
    text = HanziConv.toTraditional(text)
    bs_text = [radical.trans_ch(x) if radical.trans_ch(x) else x for x in text]
    return bs_text



def get_new_sentence(path):
    with open(path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    lines = [line.strip().split('\t') for line in lines]
    ids = [line[0] for line in lines]
    ps = [line[1] for line in lines]
    qs = [line[2] for line in lines]
    labels = [line[-1] for line in lines]
    ps = text_clean(ps)
    qs = text_clean(qs)

    p_char = [split_char(pc) for pc in ps]
    q_char = [split_char(qc) for qc in qs]
    pc_pinyin = [pinyin_char(pc) for pc in ps]
    qc_pinyin = [pinyin_char(qc) for qc in qs]
    p_bs = [get_radical(pc) for pc in ps]
    q_bs = [get_radical(qc) for qc in qs]
    p_seg = [jieba.lcut(p) for p in ps]
    q_seg = [jieba.lcut(q) for q in qs]

    x = ''
    py = ''
    bs = ''
    for idx, p, pc, q, qc, label in zip(ids, p_seg, p_char, q_seg, q_char, labels):
        p = ' '.join(p)
        q = ' '.join(q)
        x += idx + '\t' + p + '\t' + pc + '\t' + q + '\t' + qc + '\t' + label + '\n'

    for ppy, qpy in zip(pc_pinyin, qc_pinyin):
        py += ppy + '\t' + qpy + '\n'
    for pb, qb in zip(p_bs, q_bs):
        pb = ' '.join(pb)
        qb = ' '.join(qb)
        bs += pb + '\t' + qb + '\n'

    x = x.strip()
    py = py.strip()
    bs = bs.strip()
    return x, py, bs


train_x, train_py, train_bs = get_new_sentence('data/BQ_train.txt')
dev_x, dev_py, dev_bs = get_new_sentence('data/BQ_dev.txt')
test_x, test_py, test_bs = get_new_sentence('data/BQ_test.txt')
with open('newdata/BQ_train.txt', 'w', encoding='utf-8') as fw:
    fw.write(train_x)
with open('newdata/BQ_dev.txt', 'w', encoding='utf-8') as fw:
    fw.write(dev_x)
with open('newdata/BQ_test.txt', 'w', encoding='utf-8') as fw:
    fw.write(test_x)

with open('pinyin/BQ_train_py.txt', 'w', encoding='utf-8') as fw:
    fw.write(train_py)
with open('pinyin/BQ_dev_py.txt', 'w', encoding='utf-8') as fw:
    fw.write(dev_py)
with open('pinyin/BQ_test_py.txt', 'w', encoding='utf-8') as fw:
    fw.write(test_py)

with open('radical/BQ_train_rad.txt', 'w', encoding='utf-8') as fw:
    fw.write(train_bs)
with open('radical/BQ_dev_rad.txt', 'w', encoding='utf-8') as fw:
    fw.write(dev_bs)
with open('radical/BQ_test_rad.txt', 'w', encoding='utf-8') as fw:
    fw.write(test_bs)






