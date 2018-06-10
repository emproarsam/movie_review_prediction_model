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
from sentimental_analysis.CleaningPreProcessingUtility import Word2VecUtility
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

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                    'labeledTrainData.tsv'), header=0, \
                       delimiter="\t", quoting=3)
    X = data.iloc[:, [0, 2]].values
    y = data.iloc[:, 1].values

    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                    'testData.tsv'), header=0, \
                       delimiter="\t", quoting=3)
    #    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
    #                                     'MylabeledTrainData.tsv'), header=0,\
    #                                        delimiter="\t", quoting=3)
    #
    #    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
    #                                     'mytestData.tsv'), header=0,\
    #                                        delimiter="\t", quoting=3)
    clean_reviews = []
    print('Cleaning and parsing the data...\n')
    for i in X:
        clean_reviews.append(" ".join(
            Word2VecUtility.review_to_wordlist(i[1], True)
        ))

    # [3] Create bag of words
    print('Creating Bag of Words.....\n')
    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=5000)
    data_features = vectorizer.fit_transform(clean_reviews)
    X_reviews = data_features.toarray()
    #    print(data_features)


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_reviews, y, test_size=0.25, random_state=0)

    #    print('the first review is: ')
    #    print(X_train["review"][0])
    #    input("Press Enter to continue...")
    #
    # [2] Clean the training data
    #    print('Download text data sets')
    #    nltk.download()


    # [4] Train the classifier -- Random Forest
    print('Training the random forest....\n')
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(X_train, y_train)
    #    clean_test_reviews = []

    # [5] Format the testing data
    #    print("Cleaning and parsing the test data")
    #    for i in X_test[:,1].tolist():
    #        clean_test_reviews.append(" ".join(
    #                KaggleWord2VecUtility.review_to_wordlist(i, True)
    #                ))
    #    test_data_features = vectorizer.fit_transform(clean_test_reviews)
    #    test_data_features = test_data_features.toarray()
    #    print(test_data_features)

    # [6] Predit reviews in testing data
    print("Predicting test labels")
    result = forest.predict(X_test)
    output = pd.DataFrame(data={"id": X.loc['id'].values, "sentiment": result})
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', \
                               'Bag_of_words_model.csv'), index=False, quoting=3)
    print("Wrote results to Bag_of_worlds_model.csv")

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, result)

    # TODO: implement roc curve plot


    import matplotlib.pyplot as plt

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Positive', 'Negative'
    positive, negative = 0, 0

    for val in result:

        if val == 1:
            positive += 1
        else:
            negative += 1
    print(positive, negative)
    plt.title('Sentimental Anlaysis')
    plt.pie([positive, negative], labels=['+ive', '-ive'], startangle=90, shadow=True, \
            explode=(0, 0.1), autopct='%.2f')
    plt.axis('equal')  # make the pie chart circular
    plt.show()

