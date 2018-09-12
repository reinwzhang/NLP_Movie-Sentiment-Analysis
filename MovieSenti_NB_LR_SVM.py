# -*- coding: utf-8 -*-
"""
Created on Sun May  6 13:30:46 2018

@author: rein9
"""
# In[1]
# =============================================================================
#                   Movie Sentiment Analysis
# This kernel is intended for an applicaiton of nltk lib to word_tokenization
# Stemming; e.g. running --> run, learner --> learn
# Lemmatization; Problems --> Problem, Applies --> Apply, Drove --> Drive
# =============================================================================
import numpy as np
import re
import pandas as pd
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# In[2]
# =============================================================================
# Read In the Data
# =============================================================================
cur_wkd = r'C:\\Users\\rein9\\ML\\NLP_Movie\\Jiuzhang_NB'
#cur_wrd = os.path.join(os.getcwd())
dir_path = os.path.join(cur_wkd,'movie_data')
train_path = os.path.join(dir_path, 'labeledTrainData.tsv')
test_path = os.path.join(dir_path, 'testData.tsv')
unlabel_path = os.path.join(dir_path, 'unlabeledTrainData.tsv')
trainData = pd.read_csv(train_path, delimiter = '\t', quoting = 3)
testData = pd.read_csv(test_path, delimiter = '\t', quoting = 3)
unlabeledTrain = pd.read_csv(unlabel_path, delimiter = '\t', quoting = 3)
print(trainData.shape, testData.shape, unlabeledTrain.shape)
y_train = trainData['sentiment']

# In[3]
def review_to_wordlist(review):
    import nltk
    #getting rid of html tag
    review = BeautifulSoup(review, "html.parser").get_text()
    #try to get rid of non-text
    review_text = re.sub("[^a-zA-Z]+", " ", review)
    #convert all to lower case
    review_text = review_text.lower()
    #tokenize
    tokens = nltk.word_tokenize(review_text)
    #gen stop word list
    stopWords = set(nltk.corpus.stopwords.words('english'))
    #remove stop words
    words = [w for w in tokens if not w in stopWords]
    #apply stemming
    review = [nltk.stem.SnowballStemmer('english').stem(w) for w in words]
    return " ".join(review)

# In[4]
# =============================================================================
# Data Cleaning and Text Processing
# =============================================================================
train_data = []
for review in trainData.review:
    train_data.append(review_to_wordlist(review))
train_data = np.array(train_data)

test_data = []
for review in testData.review:
    test_data.append(review_to_wordlist(review))
test_data = np.array(test_data)
print('Training Dim:', train_data.shape, 'Test Dim:', test_data.shape)
# In[5]
# =============================================================================
#Transforming
#TF-IDF (Term Frequency - Inverse Document Frequency)
#can be represented tf(d,t) X idf(t).
#TF-IDF uses the method diminishing the weight (importance) of words appeared in many documents in common;
#considered them incapable of discerning the documents;
#Rather than simply counting the frequency of words as CountVectorizer does.
#The outcome matrix consists of each document (row) and each word (column)
#and the importance (weight) computed by tf * idf (values of the matrix).
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer
# a simple counter
vectorizer = CountVectorizer()
tfidf = TfidfVectorizer()
trainData_count = vectorizer.fit_transform(train_data)
testData_count = vectorizer.transform(test_data)
# In[6]:
# =============================================================================
# log probablity
# without setting the max feature number the total feature number comes to "3925964", way too big
# TFIDF transform with smooth idf (log(A+1))
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
       ngram_range=(1, 3),
       use_idf=True,
       stop_words = 'english',
       max_features = 50000,
       smooth_idf=True)
trainData_count_tf = tfidf.fit_transform(train_data)
testData_count_tf  = tfidf.transform(test_data)
# In[7]
# =============================================================================
# TFIDF tranform with sublinear_tf(1+log(A)) and max of 50,000 features
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_sub = TfidfVectorizer(
       ngram_range=(1, 3),
       use_idf=True,
       stop_words = 'english',
       max_features = 50000,
       sublinear_tf=True)
trainData_count_ltf = tfidf_sub.fit_transform(train_data)
testData_count_ltf  = tfidf_sub.transform(test_data)
# In[8]
# =============================================================================
# Get the tfidf sum
# =============================================================================
#word_freq_df = pd.DataFrame({'term': tfidf.get_feature_names(), 'tfidf': trainData_count_tf.toarray().sum(axis=0)})
word_freq_df = pd.DataFrame({'term': tfidf_sub.get_feature_names(), 'tfidf': trainData_count_ltf.toarray().sum(axis=0)})
plt.plot(word_freq_df.tfidf)
plt.show()

word_freq_df_sort = word_freq_df.sort_values(by=['tfidf'], ascending=False)
word_freq_df_sort.head()
print("Training Dim:", trainData_count_ltf.shape, 'Test Dim: ', testData_count_ltf.shape)

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

wordcloud(' '.join(word_freq_df_sort['term']))
# In[10]
# =============================================================================
# Modeling
# =============================================================================
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

# introducing 5 KFolds
kfold = StratifiedKFold(n_splits = 5, random_state = 42)

# In[11]
#1. MultinominalNB
mnb = MultinomialNB()
mnb_cv = GridSearchCV(mnb, param_grid = {'alpha': [0.1]}, scoring='accuracy',verbose = 1, cv = kfold, n_jobs= 1)
mnb_cv.fit(trainData_count_ltf, y_train)
mnb_best_par = mnb_cv.best_estimator_
print('Grid CV Search Best params:', mnb_cv.best_params_)

#prediction
pred_mnb = mnb_cv.predict(testData_count_ltf)
print(mnb_cv.best_score_)
#0.87336

df = pd.DataFrame({'id': testData.id, 'pred': pred_mnb})
pred_path = os.path.join(dir_path, 'MNB.csv')
df.to_csv(pred_path, index= False, header = True)
# In[12]
#2. BernoullliNB
bnb = BernoulliNB()
bnb_cv = GridSearchCV(bnb, param_grid = {'alpha': [0.001],
                                         'binarize': [0.001]}, verbose = 1, cv = kfold, n_jobs = 1, scoring = "roc_auc")
bnb_cv.fit(trainData_count_ltf, y_train)
bnb_best_par = bnb_cv.best_estimator_
print(bnb_cv.best_params_)
#{'alpha': 0.001, 'binarize': 0.001}
#[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.8s finished

#prediction
pred_bnb = bnb_cv.predict(testData_count_ltf)
print(bnb_cv.best_score_)
#0.9311

df = pd.DataFrame({'id': testData.id, 'pred': pred_bnb})
pred_path = os.path.join(dir_path, 'BNB.csv')
df.to_csv(pred_path, index= False, header = True)
# In[13]
#3. Linear SVC
lsvc = LinearSVC(random_state = 42)
lsvc_grid = {'C': [1, 0.2, 0.01],
              'loss': ['squared_hinge'],
              'penalty': ['l2'],
              'class_weight': [{1:4}]}
lsvc_cv = GridSearchCV(lsvc, param_grid = [lsvc_grid], verbose = 1, cv = kfold, n_jobs=1, scoring = 'roc_auc')
lsvc_cv.fit(trainData_count_ltf, y_train)
lsvc_best_par = lsvc_cv.best_estimator_
print(lsvc_cv.best_params_)
#{'C': 0.2, 'class_weight': {1: 4}, 'loss': 'squared_hinge', 'penalty': 'l2'}

#prediction
pred_lsvc = lsvc_cv.predict(testData_count_ltf)
print(lsvc_cv.best_score_)
#0.9588

df = pd.DataFrame({'id': testData.id, 'pred': pred_lsvc})
pred_path = os.path.join(dir_path, 'Linear_SVC.csv')
df.to_csv(pred_path, index= False, header = True)

# In[]:
lr = LogisticRegression(random_state = 2018)
lr_grid = {'solver': ['sag', 'lbfgs','newton-cg'],
           'C': [6,4,1,0.5],
           'multi_class' : ['multinomial'],
           'penalty': ['l2']}
# dual is only implemented with l2 in liblinear solver
# in the latest sklearn lr, l1 can be combined with SAGA solver
lr_cv = GridSearchCV(lr, param_grid = [lr_grid], verbose = 1, cv = kfold, n_jobs=1, scoring = 'roc_auc')
lr_cv.fit(trainData_count_ltf, y_train)
lr_best_par = lr_cv.best_estimator_
print(lr_cv.best_params_)
#[Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed:  3.4min finished
{'C': 4, 'multi_class': 'multinomial', 'penalty': 'l2', 'solver': 'sag'}

#prediction
pred_lr = lr_cv.predict(testData_count_ltf)
print(lr_cv.best_score_)
#0.9589

df = pd.DataFrame({'id': testData.id, 'pred': pred_lr})
pred_path = os.path.join(dir_path, 'LR.csv')
df.to_csv(pred_path, index= False, header = True)

# In[14]:
mlp = MLPClassifier(random_state = 42)
mlp_grid= {'hidden_layer_sizes': [5],
           'activation': ['relu'],
           'solver': ['adam'],
           'alpha' : [0.3, 0.1],
           'learning_rate': ['constant'],
           'max_iter': [1000],
           'batch_size' :[100]
           }
mlp_cv = GridSearchCV(mlp, param_grid = mlp_grid, verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc')
mlp_cv.fit(trainData_count_ltf, y_train)
mlp_best_par = mlp_cv.best_estimator_
print(mlp_cv.best_params_)

#prediction
pred_mlp = mlp_cv.predict(testData_count_ltf)
print(mlp_cv.best_score_)
#0.9533

df = pd.DataFrame({'id': testData.id, 'pred': pred_mlp})
pred_path = os.path.join(dir_path, 'MLP.csv')
df.to_csv(pred_path, index= False, header = True)
