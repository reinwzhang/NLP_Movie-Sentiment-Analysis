# NLP_Movie-Sentiment-Analysis
Bag of Words Meets Bags of Popcorn
The labelled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labelled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.

Input Data:
https://github.com/jiuzhangjiangzuo/naive_bayes/tree/master/movie/input
labeledTrainData - The labelled training set. The file is tab-delimited and has a header row followed by 25,000 rows containing an id, sentiment, and text for each review.
testData - The test set. The tab-delimited file has a header row followed by 25,000 rows containing an id and text for each review. Your task is to predict the sentiment for each one.
unlabeledTrainData - An extra training set with no labels. The tab-delimited file has a header row followed by 50,000 rows containing an id and text for each review.
sampleSubmission - A comma-delimited sample submission file in the correct format.

Stanford Glove Dict:
https://github.com/stanfordnlp/GloVe

Description:
MovieSenti_NB_LR_SVM.py
  Presented with ML models as Naive Bayes, Linear Support Vector Machine, Logistic Regression and MLPClassifier for input comments tokenized with TF-IDF.
  
MovieSenti_NLP_solution.py
  Presented with a couple RNN model to fit input comments tokenized with Keras Preprocessing and Word2Vec with Stanford GloVe Dict.
