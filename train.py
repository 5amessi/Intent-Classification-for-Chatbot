from docutils.nodes import header
from keras.layers import GRU
from numpy.core.multiarray import dtype
from pygments.lexer import words
from nltk import *
from nltk.tokenize import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.preprocessing import sequence,text
from keras import *
from keras.layers import *
import keras
import os
import string
from nltk.corpus import stopwords
from keras import backend as K
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = "gp"
Wk = word_tokenize
LEM = stem.WordNetLemmatizer()
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 20
HIDDEN_LAYER_SIZE = 200
LAYERS = 1

def embedding(data):
    embeddings_index = {}
    f = open(os.path.join('glove.6B.50d.txt'))
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    for sent in data:
        for word in sent:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                embeddings_index[word] = np.random.rand(EMBEDDING_DIM)
                print("iiiiiiiiiiiiiiiiiiiiiiii  = ",word, " sss ")
    length = len(embeddings_index)+1
    ind_to_word = []
    word_to_ind = {}
    ind_to_vec = np.random.rand(length,EMBEDDING_DIM)
    ind_to_vec[0] = np.zeros(EMBEDDING_DIM)
    ind = 1
    for word , vec in embeddings_index.items():
        ind_to_word.append(word)
        ind_to_vec[ind] = vec
        word_to_ind[word] = ind
        ind += 1
    return ind_to_vec , word_to_ind , ind_to_word

def seq_data(data,word_to_ind):
    temp = data.apply(lambda row: [word_to_ind[i] for i in row])
    return temp

def read_data():
    #message,food,recharge,support,reminders,travel,nearby,movies,casual,other
    data = pd.read_csv('kvret_dataset')
    x = data['message']
    y = data['intent']
    temp_unique_label = y.unique()
    unique_label = {}
    for i , label in enumerate(temp_unique_label):
        unique_label[i] = label
        y= y.replace((label),(i))
    y = keras.utils.to_categorical(y,len(unique_label))
    x = preprocess(x)
    return x ,y,unique_label

def preprocess(data,stem = False):
    #stop = stopwords.words('english')
    stop = list(string.punctuation)
    #stop = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    print(stop)
    tokenizer = TreebankWordTokenizer()
    p_stemmer = PorterStemmer()
    list_of_X = data.apply(lambda row: row.lower())
    # list_of_X = list_of_X.apply(lambda row: [i for i in (row.split())])
    list_of_X = list_of_X.apply(lambda row: tokenizer.tokenize(row))
    #list_of_X = list_of_X.apply(lambda row: [LEM.lemmatize(i) for i in row])
    #list_of_X = list_of_X.apply(lambda row: [p_stemmer.stem(i) for i in row])
    list_of_X = list_of_X.apply(lambda row: [i for i in row if i not in stop])
    #list_of_X = list_of_X.apply(lambda row: str(row))
    return list_of_X

x , y ,unique_label = read_data()

ind_to_vec,word_to_ind,ind_to_word = embedding(x)
x = seq_data(x , word_to_ind)
x = keras.preprocessing.sequence.pad_sequences(x,MAX_SEQUENCE_LENGTH,padding='pre',truncating='post',value=0)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

def KERAS():
    model = Sequential()
    model.add(Embedding(input_dim=len(ind_to_vec), output_dim=EMBEDDING_DIM,
                      weights=[ind_to_vec], input_length=MAX_SEQUENCE_LENGTH))

    model.add(GRU(128,unroll=True))
    model.add(Dense(32,activation='tanh'))
    model.add(Dense(len(unique_label),activation = 'softmax'))

    model.compile(keras.optimizers.adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_x, train_y,epochs=2,batch_size=128,verbose=1,validation_data=(test_x,test_y))
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

KERAS()
def export_model(saver,input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")
#export_model(tf.train.Saver(),["embedding_1_input"], "dense_2/Softmax")

#Train on 1939 samples, validate on 485 samples
