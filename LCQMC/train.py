import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from stats_graph import stats_graph
from keras import backend as K
from keras.engine.topology import Layer

# 指定第一块GPU可用 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#++++++++++++++++++++++++++++++++++++000000++++++++++++++++++++
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#tf.compat.v1.disable_eager_execution()
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
#sess = tf.compat.v1.Session(config=config)
from keras.utils import multi_gpu_model
from keras.utils import plot_model
import data_helper
from keras.layers import Embedding, Input, Bidirectional, LSTM, Concatenate, Add, Dropout, Dense, \
    BatchNormalization, Lambda, Activation, multiply, concatenate, Flatten, add, Dot,Permute, GlobalAveragePooling1D,MaxPooling1D, GlobalMaxPooling1D, TimeDistributed
from keras.models import Model
import keras.backend as K
from keras.callbacks import *
from tensorflow.python.ops.nn import softmax



input_dim = data_helper.MAX_SEQUENCE_LENGTH
EMBDIM = data_helper.EMBDIM
embedding_matrix = data_helper.load_pickle('embedding_matrix.pkl')
py_embedding_matrix = data_helper.load_pickle('py_embedding_matrix.pkl')
rad_embedding_matrix = data_helper.load_pickle('rad_embedding_matrix.pkl')
model_data = data_helper.load_pickle('model_data.pkl')
embedding_layer = Embedding(embedding_matrix.shape[0], EMBDIM, weights = [embedding_matrix], trainable=False)
py_embedding_layer = Embedding(py_embedding_matrix.shape[0], 70, weights = [py_embedding_matrix], trainable=False)
rad_embedding_layer = Embedding(rad_embedding_matrix.shape[0], 70, weights = [rad_embedding_matrix], trainable=False)
def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in2_aligned = Dot(axes=1)([w_att_1, input_1])
    in1_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def inter_attention(input_1):
    attention = Dot(axes=-1)([input_1, input_1])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in1_aligned = add([input_1, in1_aligned])
    return in1_aligned

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall

def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision



def matching(p,q):
    abs_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([p, q])
    min_diff = Lambda(lambda x: x[0] - x[1])([p, q])
    #cos_diff = Lambda(lambda x: K.cos(x[0] - x[1]))([p, q])
    multi_diff = multiply([p, q])
    all_diff = concatenate([abs_diff, multi_diff, min_diff])
    return all_diff

def base_model(input_shape):
    w_input = Input(shape = input_shape)
    c_input = Input(shape = input_shape)
    py_input = Input(shape = input_shape)
    rad_input = Input(shape = input_shape)
    w_embedding = embedding_layer(w_input)
    c_embedding = embedding_layer(c_input)
    py_embedding = py_embedding_layer(py_input)
    rad_embedding = rad_embedding_layer(rad_input)


    w_l = Bidirectional(LSTM(300,return_sequences='True', dropout=0), merge_mode = 'sum')(w_embedding)

    c_l = Bidirectional(LSTM(300,return_sequences='True', dropout=0), merge_mode = 'sum')(c_embedding)

    py_l = Bidirectional(LSTM(70,return_sequences='True', dropout=0.25), merge_mode = 'sum')(py_embedding)

    rad_l = Bidirectional(LSTM(70,return_sequences='True', dropout=0.25), merge_mode = 'sum')(rad_embedding)


    pyd = Dense(300)(py_l)
    radd = Dense(300)(rad_l)

    cpy, pyc = align(c_l, pyd)
    wpy, pyw= align(w_l, pyd)
    crad, radc = align(c_l, radd)
    wrad, radw = align(w_l, radd)

    cpy_aligned, wpy_aligned = align(cpy, wpy)
    crad_aligned, wrad_aligned = align(crad, wrad)

    pyc_aligned, pyw_aligned = align(pyc, pyw)
    radc_aligned, radw_aligned = align(radc, radw)


    w = concatenate([w_l, wpy, wrad, wpy_aligned, wrad_aligned])
    c = concatenate([c_l, cpy, crad, cpy_aligned, crad_aligned])
    py = concatenate([pyd, pyc, pyw, pyc_aligned, pyw_aligned])
    rad = concatenate([radd, radc, radw, radc_aligned, radw_aligned])

    inter_c = concatenate([cpy, pyc, crad, radc])
    inter_w = concatenate([wpy, pyw, wrad, radw])






    model = Model([w_input, c_input, py_input, rad_input],[w, c, py, rad, inter_c, inter_w], name = 'base_model')
    model.summary()
    return model


def siamese_model():
    input_shape = (input_dim,)
    input_pw = Input(shape = input_shape)
    input_pc = Input(shape = input_shape)
    input_qw = Input(shape = input_shape)
    input_qc = Input(shape = input_shape)
    input_ppy = Input(shape = input_shape)
    input_qpy = Input(shape = input_shape)
    input_prad = Input(shape = input_shape)
    input_qrad = Input(shape = input_shape)


    base_net = base_model(input_shape)

    pw, pc, ppy, prad, p_inter_c, p_inter_w = base_net([input_pw, input_pc, input_ppy, input_prad])
    qw, qc, qpy, qrad, q_inter_c, q_inter_w = base_net([input_qw, input_qc, input_qpy, input_qrad])

    p_inter_c_align, q_inter_c_align = align(p_inter_c, q_inter_c)
    p_inter_w_align, q_inter_w_align = align(p_inter_w, q_inter_w)

    p_inter_c = inter_attention(concatenate([p_inter_c, p_inter_c_align]))
    q_inter_c = inter_attention(concatenate([q_inter_c, q_inter_c_align]))
    p_inter_w = inter_attention(concatenate([p_inter_w, p_inter_w_align]))
    q_inter_w = inter_attention(concatenate([q_inter_w, q_inter_w_align]))





    pw = inter_attention(pw)
    qw = inter_attention(qw)
    pc = inter_attention(pc)
    qc = inter_attention(qc)

    ppy = inter_attention(ppy)
    qpy = inter_attention(qpy)
    prad = inter_attention(prad)
    qrad = inter_attention(qrad)



    pw = add([GlobalMaxPooling1D()(pw), GlobalAveragePooling1D()(pw)])
    qw = add([GlobalMaxPooling1D()(qw), GlobalAveragePooling1D()(qw)])
    pc = add([GlobalMaxPooling1D()(pc), GlobalAveragePooling1D()(pc)])
    qc = add([GlobalMaxPooling1D()(qc), GlobalAveragePooling1D()(qc)])

    ppy = add([GlobalMaxPooling1D()(ppy), GlobalAveragePooling1D()(ppy)])
    qpy = add([GlobalMaxPooling1D()(qpy), GlobalAveragePooling1D()(qpy)])
    prad = add([GlobalMaxPooling1D()(prad), GlobalAveragePooling1D()(prad)])
    qrad = add([GlobalMaxPooling1D()(qrad), GlobalAveragePooling1D()(qrad)])

    p_inter_c = add([GlobalMaxPooling1D()(p_inter_c), GlobalAveragePooling1D()(p_inter_c)])
    q_inter_c = add([GlobalMaxPooling1D()(q_inter_c), GlobalAveragePooling1D()(q_inter_c)])
    p_inter_w = add([GlobalMaxPooling1D()(p_inter_w), GlobalAveragePooling1D()(p_inter_w)])
    q_inter_w = add([GlobalMaxPooling1D()(q_inter_w), GlobalAveragePooling1D()(q_inter_w)])





    
    all_diff1 = matching(pw,qw)
    all_diff2 = matching(pc,qc)
    all_diff3 = matching(ppy, qpy)
    all_diff4 = matching(prad, qrad)
    all_diff5 = matching(p_inter_c, q_inter_c)
    all_diff6 = matching(p_inter_w, q_inter_w)



    all_diff = concatenate([all_diff1, all_diff2, all_diff3, all_diff4, all_diff5, all_diff6])

    






    all_diff = Dropout(0.6)(all_diff)

    similarity = Dense(600)(all_diff)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('relu')(similarity)
    similarity = Dense(600)(similarity)
    similarity = Dropout(0.6)(similarity)
    similarity = Activation('relu')(similarity)
    #
    similarity = Dense(1)(similarity)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('sigmoid')(similarity)
    model = Model([input_pw, input_pc, input_qw, input_qc, input_ppy, input_qpy, input_prad, input_qrad], [similarity])

    model.summary()

    margin = 0.75
    theta = lambda t: (K.sign(t) + 1.) / 2.

    def loss(y_true, y_pred):
        return -(1 - theta(y_true - margin) * theta(y_pred - margin) - theta(1 - margin - y_true) * theta(
            1 - margin - y_pred)) * (y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy', precision, recall, f1_score])

    return model

    


def train():
    data = data_helper.load_pickle('model_data.pkl')
    train_pw = data['train_pw']
    train_pc = data['train_pc']
    train_qw = data['train_qw']
    train_qc = data['train_qc']
    train_ppy = data['train_ppy']
    train_qpy = data['train_qpy']
    train_prad = data['train_prad']
    train_qrad = data['train_qrad']
    train_y = data['train_label']

    dev_pw = data['dev_pw']
    dev_pc = data['dev_pc']
    dev_qw = data['dev_qw']
    dev_qc = data['dev_qc']
    dev_ppy = data['dev_ppy']
    dev_qpy = data['dev_qpy']
    dev_prad = data['dev_prad']
    dev_qrad = data['dev_qrad']
    dev_y = data['dev_label']

    test_pw = data['test_pw']
    test_pc = data['test_pc']
    test_qw = data['test_qw']
    test_qc = data['test_qc']
    test_ppy = data['test_ppy']
    test_qpy = data['test_qpy']
    test_prad = data['test_prad']
    test_qrad = data['test_qrad']

    test_y = data['test_label']
  
  
    #tensorboard_path = 'tensorboard'
    model = siamese_model()
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)
    model_path = 'nni21.best.h5'
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    #tensorboard = TensorBoard(log_dir=tensorboard_path)
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max')
    callbackslist = [checkpoint,earlystopping, reduce_lr]

    history = model.fit([train_pw, train_pc, train_qw, train_qc, train_ppy, train_qpy, train_prad, train_qrad], train_y,
                        batch_size=512,
                        epochs=200,
                        validation_data=([dev_pw, dev_pc, dev_qw, dev_qc, dev_ppy, dev_qpy, dev_prad, dev_qrad], dev_y),
                        callbacks=callbackslist)
    '''
    ## Add graphs here
    import matplotlib.pyplot as plt

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])   
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss','train accuracy', 'val accuracy','train precision', 'val precision','train recall', 'val recall','train f1_score', 'val f1_score'], loc=3,
               bbox_to_anchor=(1.05,0),borderaxespad=0)
    pic = plt.gcf()
    pic.savefig ('pic.eps',format = 'eps',dpi=1000)
    plt.show()
    '''
    print('asd')
    loss, accuracy, precision, recall, f1_score = model.evaluate([test_pw, test_pc, test_qw, test_qc, test_ppy, test_qpy, test_prad, test_qrad], test_y, verbose=1, batch_size=256)
    print("model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (
    loss, accuracy, precision, recall, f1_score))

    x = "model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (
    loss, accuracy, precision, recall, f1_score)
    model = siamese_model()
    model.load_weights(model_path)
    loss, accuracy, precision, recall, f1_score = model.evaluate([test_pw, test_pc, test_qw, test_qc, test_ppy, test_qpy, test_prad, test_qrad], test_y, verbose=1, batch_size=256)
    y = "best model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (loss, accuracy, precision, recall, f1_score)

    with open('nni.txt', 'a') as f:

        f.write(x)
        f.write('\n')
        f.write(y)
        f.write('\n')


    


if __name__ == '__main__':

    train()