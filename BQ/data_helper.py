# -*- coding: utf-8 -*-

import re
import pickle
import numpy as np
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# 读取数据
def read_data(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = f.readlines()


    data = [re.split('\t', i) for i in data]
    pw = [i[1] for i in data]
    
    pc = [i[2] for i in data]
    qw = [i[3] for i in data]
    
    qc = [i[4] for i in data]
    label = [int(i[5]) for i in data]
    return pw, pc, qw, qc, label

def read_pinyin_radical(py_file_path, rad_file_path):
    with open(py_file_path, encoding='utf-8') as fr:
        data = fr.readlines()

    data = [re.split('\t', i) for i in data]
    ppy = [i[0].strip() for i in data]
    qpy = [i[1].strip() for i in data]
    with open(rad_file_path, encoding='utf-8') as fr:
        data = fr.readlines()

    data = [re.split('\t', i) for i in data]
    prad = [i[0].strip() for i in data]
    qrad = [i[1].strip() for i in data]

    return ppy, qpy, prad, qrad





train_pw, train_pc, train_qw, train_qc, train_label = read_data('newdata/BQ_train.txt')
dev_pw, dev_pc, dev_qw, dev_qc, dev_label = read_data('newdata/BQ_dev.txt')
test_pw, test_pc, test_qw, test_qc, test_label = read_data('newdata/BQ_test.txt')


train_ppy, train_qpy, train_prad, train_qrad = read_pinyin_radical('pinyin/BQ_train_py.txt', 'radical/BQ_train_rad.txt')
dev_ppy, dev_qpy, dev_prad, dev_qrad = read_pinyin_radical('pinyin/BQ_dev_py.txt', 'radical/BQ_dev_rad.txt')
test_ppy, test_qpy, test_prad, test_qrad = read_pinyin_radical('pinyin/BQ_test_py.txt', 'radical/BQ_test_rad.txt')







# 构造训练word2vec的语料库
corpus = train_pw + train_pc + train_qw + train_qc + test_pw + test_pc + test_qw + test_qc + dev_pw + dev_pc + dev_qw + dev_qc
w2vcorpus = train_pw + train_pc + train_qw + train_qc
w2v_corpus = [i.split() for i in w2vcorpus]
#词表
word_set = set(' '.join(corpus).split())


MAX_SEQUENCE_LENGTH = 30  # sequence最大长度为30个词
EMBDIM = 300  # 词向量为300维

# 训练word2vec模型
w2v_model = models.Word2Vec(w2v_corpus, size=EMBDIM, window=5, min_count=1, sg=1, workers=4, seed=1234, iter=25)
w2v_model.save('w2v_model.pkl')

tokenizer = Tokenizer(num_words=len(word_set))
tokenizer.fit_on_texts(corpus)
#生成的词表长度，等于num_words=len(word_set)
L = len(tokenizer.word_index)

train_pw = tokenizer.texts_to_sequences(train_pw)
train_pc = tokenizer.texts_to_sequences(train_pc)

train_qw = tokenizer.texts_to_sequences(train_qw)
train_qc = tokenizer.texts_to_sequences(train_qc)


test_pw = tokenizer.texts_to_sequences(test_pw)
test_pc = tokenizer.texts_to_sequences(test_pc)

test_qw = tokenizer.texts_to_sequences(test_qw)
test_qc = tokenizer.texts_to_sequences(test_qc)


dev_pw = tokenizer.texts_to_sequences(dev_pw)
dev_pc = tokenizer.texts_to_sequences(dev_pc)

dev_qw = tokenizer.texts_to_sequences(dev_qw)
dev_qc = tokenizer.texts_to_sequences(dev_qc)


train_pad_pw = pad_sequences(train_pw, maxlen=MAX_SEQUENCE_LENGTH)
train_pad_pc = pad_sequences(train_pc, maxlen=MAX_SEQUENCE_LENGTH)
train_pad_qw = pad_sequences(train_qw, maxlen=MAX_SEQUENCE_LENGTH)
train_pad_qc = pad_sequences(train_qc, maxlen=MAX_SEQUENCE_LENGTH)

test_pad_pw = pad_sequences(test_pw, maxlen=MAX_SEQUENCE_LENGTH)
test_pad_pc = pad_sequences(test_pc, maxlen=MAX_SEQUENCE_LENGTH)
test_pad_qw = pad_sequences(test_qw, maxlen=MAX_SEQUENCE_LENGTH)
test_pad_qc = pad_sequences(test_qc, maxlen=MAX_SEQUENCE_LENGTH)

dev_pad_pw = pad_sequences(dev_pw, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad_pc = pad_sequences(dev_pc, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad_qw = pad_sequences(dev_qw, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad_qc = pad_sequences(dev_qc, maxlen=MAX_SEQUENCE_LENGTH)

embedding_matrix = np.zeros([len(tokenizer.word_index) + 1, EMBDIM])

for word, idx in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[idx, :] = w2v_model.wv[word]

def save_pickle(fileobj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(fileobj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        fileobj = pickle.load(f)
    return fileobj




py_corpus = train_ppy + train_qpy + dev_ppy + dev_qpy + test_ppy + test_qpy
rad_corpus = train_prad + train_qrad + dev_prad + dev_qrad + test_prad + test_qrad
py_w2v_corpus = train_ppy + train_qpy
rad_w2v_corpus = train_prad + train_qrad
py_w2v_corpus = [i.split() for i in py_w2v_corpus]
rad_w2v_corpus = [i.split() for i in rad_w2v_corpus]
#词表
py_w2v_model = models.Word2Vec(py_w2v_corpus, size=70, window=5, min_count=1, sg=1, workers=4, seed=1234, iter=25)
rad_w2v_model = models.Word2Vec(rad_w2v_corpus, size=70, window=5, min_count=1, sg=1, workers=4, seed=1234, iter=25)
py_w2v_model.save('py_w2v_model.pkl')
rad_w2v_model.save('rad_w2v_model.pkl')
py_word_set = set(' '.join(py_corpus).split())
rad_word_set = set(' '.join(rad_corpus).split())
py_tokenizer = Tokenizer(num_words=len(py_word_set))
py_tokenizer.fit_on_texts(py_corpus)
rad_tokenizer = Tokenizer(num_words=len(rad_word_set))
rad_tokenizer.fit_on_texts(rad_corpus)

train_ppy = py_tokenizer.texts_to_sequences(train_ppy)
train_qpy = py_tokenizer.texts_to_sequences(train_qpy)
train_pad_ppy = pad_sequences(train_ppy, maxlen=MAX_SEQUENCE_LENGTH)
train_pad_qpy = pad_sequences(train_qpy, maxlen=MAX_SEQUENCE_LENGTH)
train_prad = rad_tokenizer.texts_to_sequences(train_prad)
train_qrad = rad_tokenizer.texts_to_sequences(train_qrad)
train_pad_prad = pad_sequences(train_prad, maxlen=MAX_SEQUENCE_LENGTH)
train_pad_qrad = pad_sequences(train_qrad, maxlen=MAX_SEQUENCE_LENGTH)

dev_ppy = py_tokenizer.texts_to_sequences(dev_ppy)
dev_qpy = py_tokenizer.texts_to_sequences(dev_qpy)
dev_pad_ppy = pad_sequences(dev_ppy, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad_qpy = pad_sequences(dev_qpy, maxlen=MAX_SEQUENCE_LENGTH)
dev_prad = rad_tokenizer.texts_to_sequences(dev_prad)
dev_qrad = rad_tokenizer.texts_to_sequences(dev_qrad)
dev_pad_prad = pad_sequences(dev_prad, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad_qrad = pad_sequences(dev_qrad, maxlen=MAX_SEQUENCE_LENGTH)


test_ppy = py_tokenizer.texts_to_sequences(test_ppy)
test_qpy = py_tokenizer.texts_to_sequences(test_qpy)
test_pad_ppy = pad_sequences(test_ppy, maxlen=MAX_SEQUENCE_LENGTH)
test_pad_qpy = pad_sequences(test_qpy, maxlen=MAX_SEQUENCE_LENGTH)
test_prad = rad_tokenizer.texts_to_sequences(test_prad)
test_qrad = rad_tokenizer.texts_to_sequences(test_qrad)
test_pad_prad = pad_sequences(test_prad, maxlen=MAX_SEQUENCE_LENGTH)
test_pad_qrad = pad_sequences(test_qrad, maxlen=MAX_SEQUENCE_LENGTH)


py_embedding_matrix = np.zeros([len(py_tokenizer.word_index) + 1, 70])

for word, idx in py_tokenizer.word_index.items():
    if word in py_w2v_model.wv:
        py_embedding_matrix[idx, :] = py_w2v_model.wv[word]

rad_embedding_matrix = np.zeros([len(rad_tokenizer.word_index) + 1, 70])

for word, idx in rad_tokenizer.word_index.items():
    if word in rad_w2v_model.wv:
        rad_embedding_matrix[idx, :] = rad_w2v_model.wv[word]


model_data = {'train_pw': train_pad_pw, 'train_pc': train_pad_pc, 'train_qw': train_pad_qw, 'train_qc': train_pad_qc,
              'train_ppy':train_pad_ppy, 'train_qpy':train_pad_qpy, 'train_prad':train_pad_prad, 'train_qrad':train_pad_qrad, 'train_label': train_label,
              'test_pw': test_pad_pw, 'test_pc': test_pad_pc, 'test_qw': test_pad_qw, 'test_qc': test_pad_qc,
              'test_ppy':test_pad_ppy, 'test_qpy':test_pad_qpy, 'test_prad':test_pad_prad, 'test_qrad':test_pad_qrad,'test_label': test_label,
              'dev_pw': dev_pad_pw, 'dev_pc': dev_pad_pc, 'dev_qw': dev_pad_qw, 'dev_qc': dev_pad_qc,
              'dev_ppy':dev_pad_ppy, 'dev_qpy':dev_pad_qpy, 'dev_prad':dev_pad_prad, 'dev_qrad':dev_pad_qrad, 'dev_label': dev_label}

save_pickle(corpus, 'corpus.pkl')
save_pickle(model_data, 'model_data.pkl')
save_pickle(embedding_matrix, 'embedding_matrix.pkl')
save_pickle(py_embedding_matrix, 'py_embedding_matrix.pkl')
save_pickle(rad_embedding_matrix, 'rad_embedding_matrix.pkl')
save_pickle(tokenizer, 'tokenizer.pkl')