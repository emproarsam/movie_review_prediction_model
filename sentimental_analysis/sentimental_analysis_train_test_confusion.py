#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:33:40 2018
sentimental anlaysis using various classifiers
@author: samarth
"""

__file__ = '/Users/samarth/Documents/anaconda projects/SentimentalAnalysisI.py'
CleaningPreProcessingUtility_PATH = '/Users/samarth/Documents/anaconda projects/'
from collections import namedtuple
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = namedtuple('model', 'type,libPath,modelClass,params')
NaiveBayes_model = model('Classifier', 'sklearn.ensemble', 'RandomForestClassifier',
                         {'n_estimators': 100, 'random_state': 0})
RandForest_model = model('Classifier', 'sklearn.naive_bayes', 'GaussianNB', {})
SVM_model = model('Classifier', 'sklearn.svm', 'SVC', {'kernel': 'linear', 'random_state': 0})

modelMap = {
    'naive': NaiveBayes_model,
    'random_forest': RandForest_model,
    'svm': SVM_model,
}


class SATrain:
    clean_documents = []

    def __init__():

        pass

    def loadData(self, inputFileName, depVars, indVars, noOfColumns):

        # [1] Read Data
        logger.info('Read Data')
        # TDOD: Train data not found
        self.data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                             inputFileName), header=0, \
                                delimiter="\t", quoting=noOfColumns)
        logger.info('Creating vectors for independent variables')
        self.X = self.data.iloc[:, indVars].values
        logger.info('Creating vectors for dependent variables')
        self.y = self.data.iloc[:, depVars].values

    def dataCleaningNProcessing(self, documentColumn, ngram=(1, 1), maxfeatures=5000):

        logger.info('Download text data sets')
        nltk.download()

        logger.info('Cleaning the data...\n')

        for i in self.X:
            self.clean_documents.append(" ".join(
                CleaningPreProcessingUtility.document_to_wordlist(i[documentColumn], True)
            ))

        logger.info('Create bag of words using TFIDF')

        #    vectorizer = CountVectorizer(analyzer="word", \
        #    vectorizer = TfidfVectorizer(analyzer="word", \
        #                                 ngram_range=(2,2), \

        # TODO: scenarios-

    #    trigrams vs unigrams vs bigrams
    #    max)features = 2.5k
    def documentVectorization(self, ngram=(1, 1), maxfeatures=5000):
        vectorizer = TfidfVectorizer(analyzer="word", \
                                     ngram_range=ngram, \
                                     #                                     tokenizer=None, \
                                     #                                     preprocessor=None, \
                                     #                                     stop_words=None, \
                                     max_features=maxfeatures)

        logger.info('Creating a pickle of vectorizer vocabulary')

        with open(vocabulary_path, 'wb') as saved_vocab:
            pickle.dump(vectorizer.vocabulary_, saved_vocab, protocol=pickle.HIGHEST_PROTOCOL)
        self.data_features = vectorizer.fit_transform(self.clean_documents)
        self.X_cleanedDocuments = self.data_features.toarray()

    def documentVectorizationFromVocab(self, saved_vocab):
        # pickle this vectorizer
        vectorizer2 = TfidfVectorizer(vocabulary=saved_vocab.vocabulary_)
        self.test_data_features = vectorizer2.fit_transform(self.clean_documents)
        self.test_data_features = self.test_data_features.toarray()
        #    print(test_data_features)

    def splitToTestTrain(self, adhocDependentCols, testProp=0.25):

        self.adhocDependentColsLen = len(adhocDependentCols)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split \
            (np.c_[self.X[:, adhocDependentCols], self.X_cleanedDocuments], self.y, test_size=testProp, random_state=0)

    def modelTraining(self, model=modelMap.get('svm')):

        import importlib
        modelClassName = model.modelClass
        modelClass = getattr(importlib.import_module(model.libPath), modelClassName)

        logger.info('Train the %s -- %s' % (modelClassName, model.type))

        self.classifier = modelClass(model.params)
        self.classifier = self.classifier.fit(self.X_train[..., self.adhocDependentColsLen:], self.y_train)
        logger.info('Creating a pickle of trained classifier')
        with open(predictor_path, 'wb') as saved_predictor:
            pickle.dump(self.classifier, saved_predictor, protocol=pickle.HIGHEST_PROTOCOL)

    def modelValidation(self, keyColumn, resultFileName, columnsCount):

        logger.info('Prediction on testing data')
        self.result = self.classifier.predict(self.X_test[..., self.adhocDependentColsLen:])
        self.output = pd.DataFrame(data={"id": self.X_test[..., keyColumn], "sentiment": self.result})
        self.output.to_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                        resultFileName), index=False, quoting=columnsCount)

        logger.info('Wrote results to %s' % (resultFileName))

        from sklearn.metrics import confusion_matrix
        logger.info('Confusion Matrix')
        cm = confusion_matrix(self.y_test, self.result)
        logger.info(str(cm))
        print(cm)
        return cm

        # TODO: implement roc curve plot

    def resultGraph(self):

        import matplotlib.pyplot as plt

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = ['+ive', '-ive']
        positive, negative = 0, 0

        for val in self.result:

            if val == 1:
                positive += 1
            else:
                negative += 1
        logger.info('No of positive responses = %s \nNo of negative responses = %s ' % (positive, negative))
        plt.title('Sentimental Anlaysis')
        plt.pie([positive, negative], labels=labels, startangle=90, shadow=True, \
                explode=(0, 0.1), autopct='%.2f')
        plt.axis('equal')  # make the pie chart circular
        plt.show()


if __name__ == "__main__":
    trainIns = SATrain()
    trainIns.loadData('labeledTrainData.tsv', 1, [0, 2], 3)
    trainIns.dataCleaningNProcessing(1)
    trainIns.documentVectorization()
    trainIns.splitToTestTrain([0])
    trainIns.modelTraining(modelMap.get('random_forest'))
    trainIns.modelValidation(0, 'TestResults25.csv', 3)

