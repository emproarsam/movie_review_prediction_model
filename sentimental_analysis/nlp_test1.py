#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:33:40 2018

@author: samarth
"""

# __file__ = '/Users/samarth/Documents/anaconda projects/SentimentalAnalysisI.py'
# Word2VecUtility_PATH = '/Users/samarth/Documents/anaconda projects/'
import sys

# if not Word2VecUtility_PATH in sys.path:
#     sys.path.append(Word2VecUtility_PATH)
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sentimental_analysis.word_2_vec_utility import Word2VecUtility
import nltk
from sklearn.naive_bayes import GaussianNB

#
# import numpy
# from keras.datasets import imdb
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence


if __name__ == "__main__":

    # [1] Read Data

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                     'labeledTrainData.tsv'), header=0, \
                        delimiter="\t", quoting=3)

    #    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
    #                                     'testData.tsv'), header=0,\
    #                                        delimiter="\t", quoting=3)
    #    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
    #                                     'myTrainData.tsv'), header=0,\
    #                                        delimiter="\t", quoting=3)

    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                    'myTestData.tsv'), header=0, \
                       delimiter="\t", quoting=3)

    print('the first review is: ')
    print(train["review"][0])
    input("Press Enter to continue...")

    # [2] Clean the training data
    print('Download text data sets')
    #    nltk.download()
    clean_train_reviews = []
    print('Cleaning and parsing the data...\n')
    for i in range(0, len(train["review"])):
        clean_train_reviews.append(" ".join(
            Word2VecUtility.review_to_wordlist(train["review"][i], True)
        ))

    # [3] Create bag of words
    print('Creating Bag of Words.....\n')
    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    print(train_data_features)

    # [4] Train the classifier -- Random Forest
    print('Training the random forest....\n')
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["sentiment"])
    # TODO: We will pickle this classifier after first execution.
    clean_test_reviews = []

    # [5] Format the testing data
    print("Cleaning and parsing the test data")
    for i in range(0, len(test["review"])):
        clean_test_reviews.append(" ".join(
            Word2VecUtility.review_to_wordlist(test["review"][i], True)
        ))
    # vectorizer2 = CountVectorizer(analyzer="word", \
    #                                 tokenizer=None, \
    #                                 preprocessor=None, \
    #                                 stop_words=None, \
    #                                 max_features=15)
    vectorizer2 = CountVectorizer(vocabulary=vectorizer.vocabulary_)
    test_data_features = vectorizer2.fit_transform(clean_test_reviews)
    #    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    print(test_data_features)

    # [6] Predit reviews in testing data
    print("Predicting test labels")
    result = forest.predict(test_data_features)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', \
                               'Bag_of_words_model.csv'), index=False, quoting=3)
    print("Wrote results to Bag_of_worlds_model.csv")

    #    from sklearn.metrics import confusion_matrix
    #    cm = confusion_matrix(train.iloc[:,1].values, output.iloc[:,1].values)
    #

    import matplotlib.pyplot as plt

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Positive', 'Negative'
    positive, negative = 0, 0

    for val in output.iloc[:, 1].values:

        if val == 1:
            positive += 1
        else:
            negative += 1
    print(positive, negative)
    plt.title('Sentimental Analaysis')
    plt.pie([positive, negative], labels=['+ve', '-ve'], startangle=90, shadow=True, \
            explode=(0, 0.1), autopct='%.2f')
    #    TODO: show count equivalent of %tages
    plt.axis('equal')  # make the pie chart circular
    plt.show()

