import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from stats_graph import stats_graph
from keras import backend as K
from keras.engine.topology import Layer

# 指定第一块GPU可用
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ++++++++++++++++++++++++++++++++++++000000++++++++++++++++++++
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# tf.compat.v1.disable_eager_execution()
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# sess = tf.compat.v1.Session(config=config)
from keras.utils import multi_gpu_model
from keras.utils import plot_model
import data_helper
from keras.layers import Embedding, Input, Bidirectional, LSTM, Concatenate, Add, Dropout, Dense, \
    BatchNormalization, Lambda, Activation, multiply, concatenate, Flatten, add, Dot, Permute, GlobalAveragePooling1D, \
    MaxPooling1D, GlobalMaxPooling1D, TimeDistributed
from keras.models import Model
import keras.backend as K
from keras.callbacks import *
from tensorflow.python.ops.nn import softmax
from train21 import siamese_model

data = data_helper.load_pickle('model_data.pkl')
test_pw = data['test_pw']
test_pc = data['test_pc']
test_qw = data['test_qw']
test_qc = data['test_qc']
test_ppy = data['test_ppy']
test_qpy = data['test_qpy']
test_prad = data['test_prad']
test_qrad = data['test_qrad']

test_y = data['test_label']
model = siamese_model()

model.load_weights('MIPRlc.best.h5')


loss, accuracy, precision, recall, f1_score = model.evaluate([test_pw, test_pc, test_qw, test_qc, test_ppy, test_qpy, test_prad, test_qrad], test_y, verbose=1, batch_size=256)
print("Test best model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (loss, accuracy, precision, recall, f1_score))