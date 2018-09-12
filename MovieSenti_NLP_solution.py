# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 23:39:24 2018

@author: rein9
"""
# In[]
# =============================================================================
#                         Movie Sentiment Analysis with NLP
# Following is a different Kernel implemented with Stanford Glove Dict for word2vec embedding
# Stemming; e.g. running --> run, learner --> learn
# Lemmatization; Problems --> Problem, Applies --> Apply, Drove --> Drive
# =============================================================================
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import langdetect
import operator

# In[2]
# =============================================================================
# Read In the Data
# =============================================================================
cur_wkd = r'C:\\Users\\rein9\\ML\\NLP_Movie\\Jiuzhang_NB'
cur_wrd = os.path.join(os.getcwd())
dir_path = os.path.join(cur_wkd, 'movie_data')
train_path = os.path.join(dir_path, 'labeledTrainData.tsv')
test_path = os.path.join(dir_path, 'testData.tsv')
unlabel_path = os.path.join(dir_path, 'unlabeledTrainData.tsv')
trainData = pd.read_csv(train_path, delimiter = '\t', quoting = 3)
testData = pd.read_csv(test_path, delimiter = '\t', quoting = 3)
unlabeledTrain = pd.read_csv(unlabel_path, delimiter = '\t', quoting = 3)
print(trainData.shape, testData.shape, unlabeledTrain.shape)

GLOVE_DIR = os.path.join(cur_wkd, '..', 'glove.6B.50d.txt')
Model_path = os.path.join(cur_wkd, 'movie_model')

# In[]:
# =============================================================================
# 1) Tokenize the train data set by joining the training and test dataset
# 2) Tokenizing with keras preprocessing
# =============================================================================
import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pandas_ml import ConfusionMatrix

from keras.models import Model, Sequential, save_model, load_model
from keras.layers import Input, Dense, Embedding, Dropout, SpatialDropout1D, concatenate
from keras.layers import LSTM, GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, History
# In[]
# =============================================================================
# Load existing tokens or initilize new tokenizer and fit on Texts
# =============================================================================
PRELOAD = False
num_units = 50 # the number of LSTM units
batch_size = 128
EMBEDDING_DIM = int(''.join([s for s in GLOVE_DIR.split('/')[-1].split('.')[-2] if s.isdigit()])) # 50
MAX_NB_WORDS = 20000
if PRELOAD:
  tokenizer=joblib.load(os.path.join(Model_path, 'tokenizer.pickle'))
  with pd.HDFStore(os.path.join(Model_path,'x_y_train_val.h5')) as h:
    X_train = h['X_train'].values
    y_train = h['y_train'].values
    X_val =h['X_val'].values
    y_val=h['y_val'].values
  model = load_model(os.path.join(Model_path,'keras_model_updated.keras'))
else:
  totalsent = trainData.review.tolist() + testData.review.tolist()
  tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
  tokenizer.fit_on_texts(totalsent)
  joblib.dump(tokenizer, 'tokenizer.pickle')

WORD_INDEX_SORTED = sorted(tokenizer.word_index.items(), key=operator.itemgetter(1))
vocabulary_size = len(WORD_INDEX_SORTED)
# In[9]
# =============================================================================
# WordCloud view
# =============================================================================
from wordcloud import WordCloud, STOPWORDS
def wordcloud(inputs, stopwords = STOPWORDS):
    clouds = WordCloud(stopwords=STOPWORDS).generate(inputs)
    plt.figure(figsize = (10,10))
    plt.imshow(clouds)
    plt.show()

wordcloud(' '.join(word for word,_ in tokenizer.word_index.items()))

# In[]:
# =============================================================================
# Encoded the training sentences
# pad with <EOS>
# if we only consider sentence length, this network will be too large, prone to overfit
# Need stem the data first
# =============================================================================
seqs = tokenizer.texts_to_sequences(trainData.review.values)
seq_len = [len(row) for row in seqs]# check length of each sentence
MAX_SEQUENCE_LENGTH = min(max(seq_len) +1, 100)# max number of words in the comment to use
X = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)
y = trainData['sentiment']
assert X.shape[0] == y.shape[0]

# split for train validation
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=2018)
with pd.HDFStore(os.path.join(Model_path,'x_y_train_val.h5')) as h:
  h['X_train'] = pd.DataFrame(X_train)
  h['y_train']=pd.DataFrame(y_train)
  h['X_val']= pd.DataFrame(X_val)
  h['y_val']=pd.DataFrame(y_val)

#In[]: tokenize and pad test_X
test_seqs = tokenizer.texts_to_sequences(testData.review.values)
test_X = pad_sequences(test_seqs,maxlen=MAX_SEQUENCE_LENGTH)
test_X_len = [len(row) for row in test_X]

# In[5]
# =============================================================================
# Glove to Vec
# =============================================================================
def load_glove_into_dict(glove_path):
    embedding_ix = {}
    with open(glove_path, encoding = 'utf8') as glove_file:
        for line in glove_file:
            val = line.split()
            word = val[0] #key
            vec = np.array(val[1:], dtype='float32') # the vec
            embedding_ix[word] = vec
    glove_file.close()
    return embedding_ix

embeddings_index = load_glove_into_dict(GLOVE_DIR)
print('Loaded %s word vectors.' % len(embeddings_index))

# In[]
# =============================================================================
# Using the glove dict to create our embedding matrix with random initialization
# for words that arent in GloVe
# use the same mean and std of embeddings the GloVe has when generating the random init
# =============================================================================
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
print('emb_mean: ', emb_mean, 'emb_std: ', emb_std)
#emb_mean:  0.020940226 emb_std:  0.64410394

# In[]
# =============================================================================
# Creating the embedding matrix
# =============================================================================
nb_words = min(MAX_NB_WORDS, vocabulary_size)
embedding_size = EMBEDDING_DIM
#embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))
embedding_matrix = np.zeros((nb_words,embedding_size))

for word,i in tokenizer.word_index.items():
    if i >= MAX_NB_WORDS:continue
    embedding_vec = embeddings_index.get(word)
    if embedding_vec is not None:
        embedding_matrix[i] = embedding_vec

# In[]
# =============================================================================
# Save check points and training history
# =============================================================================
filepath = os.path.join(Model_path, 'imp-{epoch:02d}-{val_acc:.2f}.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
csv_logger = CSVLogger(os.path.join(Model_path, 'training_history.csv'))
history=History()
callbacks_list = [checkpoint, history, csv_logger]

# In[]
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

# In[]
# =============================================================================
#           Model 1) Simple
# Embedding --> LSTM --> Dense(underfit)
# =============================================================================
embedding_layer = Embedding(embedding_matrix.shape[0],
                embedding_matrix.shape[1],
                weights=[embedding_matrix],
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=False)
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(num_units))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
hist = model.fit(X_train,
          y_train,
          batch_size = batch_size,
          epochs = 4,
          validation_data=(X_val,y_val),
          callbacks=[RocAuc], verbose=2)
# =============================================================================
 #Epoch 4/4
 # - 16s - loss: 0.4976 - acc: 0.7610 - val_loss: 0.4995 - val_acc: 0.7484
 # ROC-AUC - epoch: 4 - score: 0.845252
# =============================================================================

preds_val=model.predict_classes(X_val)
cm=ConfusionMatrix(y_val,preds_val.ravel())
cm.plot(backend='seaborn', normalized=True)
plt.title('Confusion Matrix for Movie Prediction')
plt.figure(figsize=(12,10))

# In[]
# =============================================================================
#               2) Bidirectional LSTM
# Embedding --> Bidirectional(LSTM) --> Pool --> Dense --> Dropout > Dense
# =============================================================================

inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
emb_1 = Embedding(MAX_NB_WORDS, embedding_size, weights = [embedding_matrix])(inp)
bid_1 = Bidirectional(LSTM(num_units, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1))(emb_1)
pool_1 = GlobalMaxPooling1D()(bid_1)
dense_1 = Dense(num_units, activation = 'relu')(pool_1)
drop_1 = Dropout(0.1)(dense_1)
dense_2 = Dense(1, activation = 'sigmoid')(drop_1)
model= Model(inputs = inp, outputs = dense_2)
model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
model.fit(X_train,
          y_train,
          batch_size = batch_size,
          epochs = 12,
          validation_data=(X_val,y_val),
          callbacks=[RocAuc], verbose=2)
# =============================================================================
#Epoch 1/12
# - 66s - loss: 0.5766 - acc: 0.6835 - val_loss: 0.4279 - val_acc: 0.8072
# ROC-AUC - epoch: 1 - score: 0.887592
#Epoch 2/12
# - 59s - loss: 0.3962 - acc: 0.8213 - val_loss: 0.3627 - val_acc: 0.8384
# ROC-AUC - epoch: 2 - score: 0.921311
#Epoch 3/12
# - 60s - loss: 0.3198 - acc: 0.8600 - val_loss: 0.3334 - val_acc: 0.8546
# ROC-AUC - epoch: 3 - score: 0.932752
#Epoch 4/12
# - 56s - loss: 0.2640 - acc: 0.8891 - val_loss: 0.3256 - val_acc: 0.8634
# ROC-AUC - epoch: 4 - score: 0.937652
#Epoch 5/12
# - 55s - loss: 0.2193 - acc: 0.9127 - val_loss: 0.3311 - val_acc: 0.8604
# ROC-AUC - epoch: 5 - score: 0.939135
#Epoch 6/12
# - 55s - loss: 0.1814 - acc: 0.9306 - val_loss: 0.3474 - val_acc: 0.8594
# ROC-AUC - epoch: 6 - score: 0.939215
#Epoch 7/12
# - 55s - loss: 0.1452 - acc: 0.9449 - val_loss: 0.3652 - val_acc: 0.8624
# ROC-AUC - epoch: 7 - score: 0.938429
#Epoch 8/12
# - 55s - loss: 0.1038 - acc: 0.9641 - val_loss: 0.4075 - val_acc: 0.8572
## ROC-AUC - epoch: 8 - score: 0.937491
#Epoch 9/12
# - 55s - loss: 0.0807 - acc: 0.9728 - val_loss: 0.4490 - val_acc: 0.8566
# ROC-AUC - epoch: 9 - score: 0.935090
#Epoch 10/12
# - 55s - loss: 0.0602 - acc: 0.9808 - val_loss: 0.4784 - val_acc: 0.8542
# ROC-AUC - epoch: 10 - score: 0.933329
#Epoch 11/12
# - 55s - loss: 0.0417 - acc: 0.9869 - val_loss: 0.5347 - val_acc: 0.8508
# ROC-AUC - epoch: 11 - score: 0.932518
#Epoch 12/12
# - 55s - loss: 0.0272 - acc: 0.9931 - val_loss: 0.5852 - val_acc: 0.8556
# ROC-AUC - epoch: 12 - score: 0.931629
# =============================================================================

print('Predicting....')
BiLSTM_pred = model.predict(test_X,batch_size=124,verbose=1)
BiLSTM_pred = [(1 if x > 0.5 else 0) for x in BiLSTM_pred[:,0]]
df = pd.DataFrame({'id': testData.id, 'pred': BiLSTM_pred})
pred_path = os.path.join(dir_path, 'BiLSTM.csv')
df.to_csv(pred_path, index= False, header = True)

# In[]
# =============================================================================
# EMB--> SpatialDropout-->BirectionalGRU--> AvgPool-->MaxPool-->Concate--> Dense
# =============================================================================
def get_model():
    inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    x = Embedding(MAX_NB_WORDS, embedding_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(50, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
model.fit(X_train,
          y_train,
          batch_size = batch_size,
          epochs = 5,
          validation_data=(X_val,y_val),
          callbacks=callbacks_list, verbose=2)

# =============================================================================
# # In[]
# Conclusion
# The best result in this notebook is acc of 0.94 and a val_acc of only 0.87
# (CountVectorizer/TF-IDF) performs better much better
# The main reason is Word2Vec suffers a loss when embedding with maxlen < length of sentense
# If trying to embed with max length of the sentence, the model will get too big to train
# Among the all methods that I attempted, the best result is 90.36% performed by Logistic Regression in TF-IDF.
# As you notice, all the best results come from Logistic Regression and the worst come from Naive Bayes.
# =============================================================================

# In[4]
# =============================================================================
#                           Tensorflow Version(too big to train)
# =============================================================================
# In[]
def data_generator(X,y,batch_size):
    while True:
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size],y[i:i+batch_size]
# In[]
# =============================================================================
# Tensorflow initializer
# =============================================================================
import tensorflow as tf
#import tflearn
num_units = 50
tf.reset_default_graph()
config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.4
sess=tf.Session(config=config)
with tf.device('/gpu:0'):
    initializer = tf.random_uniform_initializer(-0.8,0.8)
    tf.get_variable_scope().set_initializer(initializer)
    X = tf.placeholder('int32', [None, None])
    y = tf.placeholder('int32',[None])
    X_len = tf.placeholder('int32', [None])
    learning_rate = tf.placeholder(tf.float32, [])

    #embedding
    embedding_encoder = tf.get_variable('embedding_encoder', [MAX_NB_WORDS, MAX_SEQUENCE_LENGTH], dtype = tf.float32)
    encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, X)

    #Build RNN cells
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    #encoder_outputs: [max_time, batch_size, num_units]
    #encoder_state: [batch_size, num_units]
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,encoder_emb_inp,
                                                       sequence_length = X_len,
                                                       time_major = False,
                                                       dtype = tf.float32)
    model_logistic = tf.layers.dense(encoder_state[0],1)
    model_pred = tf.nn.sigmoid(model_logistic) #activation
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y,tf.float32),
                                                   logits=tf.reshape(model_logistic, (-1,)))
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
# In[]
# =============================================================================
# RUN INFERENCE
# =============================================================================
sess.run(tf.global_variables_initializer())
from utils import ProgressBar
losses = []
beginning_lr = 0.1
gen = data_generator(X_train, y_train, batch_size)
for one_epoch in range(1,10):
    pb = ProgressBar(worksum=MAX_SEQUENCE_LENGTH)
    pb.startjob()
    for one_batch in range(0, len(X_train), batch_size):
        batch_x, batch_y = gen.__next__()
        batch_x_len = np.asarray([len(x) for x in batch_x])
        batch_lr = beginning_lr

        _,batch_loss = sess.run([optimizer, loss],
                                feed_dict = {
                                        X:batch_x,
                                        y:batch_y,
                                        X_len:batch_x_len,
                                        learning_rate:batch_lr})
    pb.info="EPOCH {} batch{} lr {} loss {}".format(one_epoch,one_batch,batch_lr,batch_loss)
    pb.complete(batch_size)
    losses.append(batch_loss)
batch_predict = sess.run(model_pred,
                         feed_dict={
                             X:test_X,
                             X_len:test_X_len})[:,0]
batch_predict = [(1 if x > 0.5 else 0) for x in batch_predict]

# In[]
# =============================================================================
# Run Batch Prediction
# =============================================================================
pd.DataFrame(losses).plot()
df = pd.DataFrame({'id': testData.id, 'pred': batch_predict})
pred_path = os.path.join(dir_path, 'TF_RNN.csv')
df.to_csv(pred_path, index= False, header = True)
