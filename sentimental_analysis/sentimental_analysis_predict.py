#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:40:43 2018
This module will predict the nature of the reviews(i.e. positive or negative)
@author: samarth
"""

__file__ = '/Users/samarth/Documents/anaconda projects/SentimentalAnalysisI.py'
CleaningPreProcessingUtility_PATH = '/Users/samarth/Documents/anaconda projects/'
import sys

if not CleaningPreProcessingUtility_PATH in sys.path:
    sys.path.append(CleaningPreProcessingUtility_PATH)
vocabulary_path = '/Users/samarth/Documents/anaconda projects/pickles/vocab.pickle'
predictor_path = '/Users/samarth/Documents/anaconda projects/pickles/predictor.pickle'

import pickle
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from CleaningPreProcessingUtility import CleaningPreProcessingUtility
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from sentimental_analysis_train_test_confusion import SATrain, modelMap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open(vocabulary_path, 'rb') as handle:
    saved_vocab = pickle.load(handle)

with open(predictor_path, 'rb') as handle:
    saved_predictor = pickle.load(handle)


class SAPrediction:
    # [1] Read Data

    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                    'myTestData.tsv'), header=0, \
                       delimiter="\t", quoting=3)

    # [2] Clean the training data
    print('Download text data sets')
    nltk.download()
    clean_test_reviews = []
    # [5] Format the testing data
    print("Cleaning and parsing the test data")
    for i in range(0, len(test["review"])):
        clean_test_reviews.append(" ".join(
            CleaningPreProcessingUtility.review_to_wordlist(test["review"][i], True)
        ))
    # pickle this vectorizer
    vectorizer2 = CountVectorizer(vocabulary=saved_vocab.vocabulary_)
    test_data_features = vectorizer2.fit_transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    print(test_data_features)

    # [6] Predit reviews in testing data
    print("Predicting test labels")
    # pickle this forest
    result = saved_predictor.predict(test_data_features)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', \
                               'Bag_of_words_model.csv'), index=False, quoting=3)
    print("Wrote results to Bag_of_worlds_model.csv")

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


def documentVectorizationFromVocab(self, ngram=(1, 1), max_features=5000):
    # pickle this vectorizer
    vectorizer2 = TfidfVectorizer(vocabulary=saved_vocab)
    self.test_data_features = vectorizer2.fit_transform(self.clean_documents)
    self.test_data_features = self.test_data_features.toarray()
    #    print(test_data_features)


def predictionFromSavedClassifier(self, classifier):
    # [6] Predit reviews in testing data
    print("Predicting test labels")
    # pickle this forest
    result = saved_predictor.predict(test_data_features)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', \
                               'Bag_of_words_model.csv'), index=False, quoting=3)
    print("Wrote results to Bag_of_worlds_model.csv")


if __name__ == "__main__":
    trainIns = SATrain()
    trainIns.loadData('myTestData.tsv', 1, [0], 2)
    trainIns.dataCleaningNProcessing(1)
    trainIns.documentVectorizationFromVocab(saved_vocab)
    trainIns.modelTraining(modelMap.get('random_forest'))
# trainIns.modelValidation(0,'TestResults25.csv',3)


