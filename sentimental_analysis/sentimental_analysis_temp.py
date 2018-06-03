#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:33:40 2018
sentimental anlaysis using various classifiers
@author: samarth
"""

from collections import namedtuple
import sys
import os
import pickle
import pandas as pd
import numpy as np
from CleaningPreProcessingUtility import CleaningPreProcessingUtility
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sentimental_analysis_predict import getSavedPredictor
import mglearn
from sentimental_analysis_train_test_confusion import pathMaker, modelMap

# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__file__ = sys.argv[0]
CleaningPreProcessingUtility_PATH = os.getcwd()

if not CleaningPreProcessingUtility_PATH in sys.path:
    sys.path.append(CleaningPreProcessingUtility_PATH)

pickles_path = os.getcwd() + '/pickles/'
vocabulary_path = pickles_path + 'vocab.pickle'

gridPath = pathMaker(pickles_path + 'grids/classifiers/SVM/predictor.pickle')

# with open(gridPath, 'rb') as handle:
#    saved_grid = pickle.load(handle)

labels = ['negative', 'positive']


class SATrain:
    clean_documents = []

    def __init__(self):
        logger.info('Downloading text data sets')
        #        nltk.download()

        pass

    def loadData(self, inputFileName, depVars, indVars, noOfColumns):

        # [1] Read Data
        logger.info('Reading Data')
        # TDOD: Train data not found
        try:
            self.data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                                 inputFileName), header=0, \
                                    delimiter="\t", quoting=noOfColumns)
        except FileNotFoundError as err:
            logger.error(str(err))
            raise err
        logger.info('Creating vectors for independent variables')
        self.X = self.data.iloc[:, indVars].values
        logger.info('Creating vectors for dependent variables')
        self.y = self.data.iloc[:, depVars].values

    def dataCleaningNProcessing(self, documentColumn, ngram=(1, 1), maxfeatures=5000):

        logger.info('Cleaning the data...\n')

        for i in self.X:
            self.clean_documents.append(" ".join(
                CleaningPreProcessingUtility.document_to_wordlist(i[documentColumn], True)
            ))

        logger.info('Creating bag of words using TFIDF')

        #    vectorizer = CountVectorizer(analyzer="word", \
        #    vectorizer = TfidfVectorizer(analyzer="word", \
        #                                 ngram_range=(2,2), \

        # TODO: scenarios-

    #    trigrams vs unigrams vs bigrams
    #    max)features = 2.5k
    def documentVectorization(self, ngram=(1, 1), maxfeatures=5000):
        self.vectorizer = TfidfVectorizer(analyzer="word", \
                                          ngram_range=ngram, \
                                          #                                     tokenizer=None, \
                                          #                                     preprocessor=None, \
                                          #                                     stop_words=None, \
                                          max_features=maxfeatures)

        logger.info('Creating a pickle of vectorizer vocabulary')
        self.data_features = self.vectorizer.fit_transform(self.clean_documents)
        with open(vocabulary_path, 'wb') as saved_vocab:
            pickle.dump(self.vectorizer.vocabulary_, saved_vocab, protocol=pickle.HIGHEST_PROTOCOL)
        self.X_cleanedDocuments = self.data_features.toarray()

    #    def documentVectorizationFromVocab(self, saved_vocab):
    #         #pickle this vectorizer
    #        vectorizer2 = TfidfVectorizer(vocabulary=saved_vocab.vocabulary_)
    #        self.test_data_features = vectorizer2.fit_transform(self.clean_documents)
    #        self.test_data_features = self.test_data_features.toarray()
    #    #    print(test_data_features)
    #

    def splitToTestTrain(self, adhocDependentCols, testProp=0.25):

        self.adhocDependentColsLen = len(adhocDependentCols)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split \
            (np.c_[self.X[:, adhocDependentCols], self.X_cleanedDocuments], self.y, test_size=testProp, random_state=0)

    def modelTraining(self, model=modelMap.get('svm'), saved=None):

        import importlib
        modelClassName = model.modelClass
        modelClass = getattr(importlib.import_module(model.libPath), modelClassName)

        logger.info('Training the %s -- %s' % (modelClassName, model.type))

        self.classifier = modelClass(**model.params)

        self.classifier = saved or self.classifier.fit(self.X_train[..., self.adhocDependentColsLen:], self.y_train)
        logger.info('Creating a pickle of trained classifier')
        with open(model.picklePath, 'wb') as saved_predictor:
            pickle.dump(self.classifier, saved_predictor, protocol=pickle.HIGHEST_PROTOCOL)

    def modelValidation(self, keyColumn, resultFileName, columnsCount):

        logger.info('Prediction on testing data')
        self.result = self.classifier.predict(self.X_test[..., self.adhocDependentColsLen:])
        self.output = pd.DataFrame(data={"id": self.X_test[..., keyColumn], "sentiment": self.result})
        self.output.to_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                        resultFileName), index=False, quoting=columnsCount)
        logger.info('Writing results to %s' % (resultFileName))
        logger.info('Confusion Matrix:\n')
        cm = confusion_matrix(self.y_test, self.result)
        logger.info(str(cm))
        print(cm)
        logger.info('Classification Metrics:\n%s' % (classification_report(self.y_test, self.result, \
                                                                           target_names=labels)))
        return cm

        # TODO: implement roc curve plot

    def resultGraph(self):

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        positive, negative = 0, 0

        for val in self.result:

            if val == 1:
                positive += 1
            else:
                negative += 1
        logger.info('No of positive responses = %s \nNo of negative responses = %s ' % (positive, negative))
        plt.title('Sentimental Anlaysis')
        plt.pie([negative, positive], labels=labels, startangle=90, shadow=True, \
                explode=(0, 0.1), autopct='%.2f', colors=['red', 'blue'], rotatelabels=True)
        plt.axis('equal')  # make the pie chart circular
        plt.show()

    def sentimentHistogram(self, top_features=40):

        coef = self.classifier.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(self.vectorizer.get_feature_names())
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()

    def moreMetrics(self, model=modelMap.get('svm')):

        global log

        print("=" * 30)
        print(model.modelClass)

        print('****Results****')
        acc = accuracy_score(self.y_test, self.result)
        print("Accuracy: {:.4%}".format(acc))
        ll = log_loss(self.y_test, self.result)
        print("Log Loss: {}".format(ll))

        log_entry = pd.DataFrame([[model.modelClass, acc * 100, ll]], columns=log_cols)
        log = log.append(log_entry)

        print("=" * 30)

        # horizontal bar plot

        import seaborn as sns
        sns.set_color_codes("muted")
        sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

        plt.xlabel('Accuracy %')
        plt.title('Classifier Accuracy')
        plt.show()

        sns.set_color_codes("muted")
        sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

        plt.xlabel('Log Loss')
        plt.title('Classifier Log Loss')
        plt.show()

    def sortedByIDF(self):

        #        self.featureNames = self.vectorizer.get_feature_names()
        #        print("Total number of features: {}\n".format(len(self.featureNames)))
        #        sorted_by_idf = np.argsort(self.vectorizer.idf_)
        #        print("Features with lowest idf:\n{}".format(self.featureNames[sorted_by_idf[:100]]))
        indices = np.argsort(self.vectorizer.idf_)[::-1]
        features = self.vectorizer.get_feature_names()
        top_n = 10
        top_features = [features[-i - 1] for i in indices[:top_n]]
        print(top_features)

    def getCrossValidationScore(self, model=modelMap.get('svm')):

        from sklearn.model_selection import cross_val_score
        import importlib
        modelClassName = model.modelClass
        modelClass = getattr(importlib.import_module(model.libPath), modelClassName)
        logger.info('Training the %s -- %s' % (modelClassName, model.type))
        _classifier = modelClass(**model.params)

        scores = cross_val_score(_classifier, self.X_train[..., self.adhocDependentColsLen:], self.y_train, cv=5)
        print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

    def GridSearch(self, model=modelMap.get('svm'), grid=None):

        import time

        st = time.time

        import importlib
        modelClassName = model.modelClass
        modelClass = getattr(importlib.import_module(model.libPath), modelClassName)
        logger.info('Training the %s -- %s' % (modelClassName, model.type))
        _classifier = modelClass(**model.params)
        self.param_grid = {'C': [10, 100], 'gamma': [0.01, 0.1]}
        self.grid = grid or GridSearchCV(estimator=_classifier, param_grid=self.param_grid, cv=5)
        self.grid.fit(self.X_train[..., self.adhocDependentColsLen:], self.y_train)
        with open(gridPath, 'wb') as saved_grid:
            pickle.dump(self.grid, saved_grid, protocol=pickle.HIGHEST_PROTOCOL)
        print("Test set score: {:.2f}".format(self.grid.score()))
        print("best cross-validation score: {:.2f}".format(self.grid.best_score_))
        print("best parameter: {}".format(self.grid.best_params_))
        print("best estimator: {}".format(self.grid.best_estimator_))
        print("it took {} seconds".format(time.time - st))

    def heatMap(self):

        # extract scores from grid
        score = self.grid.cv_results_['mean_test_score'].shape(-1, 3).T
        # visualize heat map
        heatmap = mglearn.tools.heatmap(
            score, xlabel="C", ylabel="ngram_range", cmap="virdis", fmt="%.3f",
            xticklabels=self.param_grid['C'],
            yticklabels=self.param_grid['gamma'])
        plt.colorbar(heatmap)

    def lDA(self):

        from sklearn.decomposition import LatentDirichletAllocation
        lda = LatentDirichletAllocation(n_topics=10, learning_method="batch",
                                        max_iter=25, random_state=0)
        doc_topics = lda.fit_transform(self.data_features)
        print(doc_topics)
        sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
        feature_names = np.array(self.vectorizer.get_feature_names())
        mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
                                   sorting=sorting, topics_per_chunk=5, n_words=10)


if __name__ == "__main__":
    trainIns = SATrain()
    trainIns.loadData('labeledTrainData.tsv', 1, [0, 2], 3)
    trainIns.dataCleaningNProcessing(1)
    trainIns.documentVectorization(ngram=(1, 2))
    trainIns.splitToTestTrain([0])
    #    trainIns.modelTraining(modelMap.get('random_forest'))
    #                           getSavedPredictor('random_forest'))
    #    trainIns.modelTraining()
    #    trainIns.modelValidation(0,'TestResults25.csv',3)
    #    trainIns.sentimentHistogram(top_features=40)
    #    trainIns.resultGraph()
    #    trainIns.moreMetrics(modelMap.get('random_forest'))
    #    trainIns.moreMetrics()
    #    trainIns.sortedByIDF()
    #    trainIns.getCrossValidationScore(modelMap.get('random_forest'))
    #    trainIns.getCrossValidationScore()
    #    trainIns.GridSearch(grid=saved_grid)
    trainIns.GridSearch()
    trainIns.heatMap()
    trainIns.lDA()

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:33:40 2018
sentimental anlaysis using various classifiers
@author: samarth
"""

from collections import namedtuple
import sys
import os
import pickle
import pandas as pd
import numpy as np
from CleaningPreProcessingUtility import CleaningPreProcessingUtility
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sentimental_analysis_predict import getSavedPredictor
import mglearn
from sentimental_analysis_train_test_confusion import pathMaker, modelMap

# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__file__ = sys.argv[0]
CleaningPreProcessingUtility_PATH = os.getcwd()

if not CleaningPreProcessingUtility_PATH in sys.path:
    sys.path.append(CleaningPreProcessingUtility_PATH)

pickles_path = os.getcwd() + '/pickles/'
vocabulary_path = pickles_path + 'vocab.pickle'

gridPath = pathMaker(pickles_path + 'grids/classifiers/SVM/predictor.pickle')

# with open(gridPath, 'rb') as handle:
#    saved_grid = pickle.load(handle)

labels = ['negative', 'positive']


class SATrain:
    clean_documents = []

    def __init__(self):
        logger.info('Downloading text data sets')
        #        nltk.download()

        pass

    def loadData(self, inputFileName, depVars, indVars, noOfColumns):

        # [1] Read Data
        logger.info('Reading Data')
        # TDOD: Train data not found
        try:
            self.data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                                 inputFileName), header=0, \
                                    delimiter="\t", quoting=noOfColumns)
        except FileNotFoundError as err:
            logger.error(str(err))
            raise err
        logger.info('Creating vectors for independent variables')
        self.X = self.data.iloc[:, indVars].values
        logger.info('Creating vectors for dependent variables')
        self.y = self.data.iloc[:, depVars].values

    def dataCleaningNProcessing(self, documentColumn, ngram=(1, 1), maxfeatures=5000):

        logger.info('Cleaning the data...\n')

        for i in self.X:
            self.clean_documents.append(" ".join(
                CleaningPreProcessingUtility.document_to_wordlist(i[documentColumn], True)
            ))

        logger.info('Creating bag of words using TFIDF')

        #    vectorizer = CountVectorizer(analyzer="word", \
        #    vectorizer = TfidfVectorizer(analyzer="word", \
        #                                 ngram_range=(2,2), \

        # TODO: scenarios-

    #    trigrams vs unigrams vs bigrams
    #    max)features = 2.5k
    def documentVectorization(self, ngram=(1, 1), maxfeatures=5000):
        self.vectorizer = TfidfVectorizer(analyzer="word", \
                                          ngram_range=ngram, \
                                          #                                     tokenizer=None, \
                                          #                                     preprocessor=None, \
                                          #                                     stop_words=None, \
                                          max_features=maxfeatures)

        logger.info('Creating a pickle of vectorizer vocabulary')
        self.data_features = self.vectorizer.fit_transform(self.clean_documents)
        with open(vocabulary_path, 'wb') as saved_vocab:
            pickle.dump(self.vectorizer.vocabulary_, saved_vocab, protocol=pickle.HIGHEST_PROTOCOL)
        self.X_cleanedDocuments = self.data_features.toarray()

    #    def documentVectorizationFromVocab(self, saved_vocab):
    #         #pickle this vectorizer
    #        vectorizer2 = TfidfVectorizer(vocabulary=saved_vocab.vocabulary_)
    #        self.test_data_features = vectorizer2.fit_transform(self.clean_documents)
    #        self.test_data_features = self.test_data_features.toarray()
    #    #    print(test_data_features)
    #

    def splitToTestTrain(self, adhocDependentCols, testProp=0.25):

        self.adhocDependentColsLen = len(adhocDependentCols)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split \
            (np.c_[self.X[:, adhocDependentCols], self.X_cleanedDocuments], self.y, test_size=testProp, random_state=0)

    def modelTraining(self, model=modelMap.get('svm'), saved=None):

        import importlib
        modelClassName = model.modelClass
        modelClass = getattr(importlib.import_module(model.libPath), modelClassName)

        logger.info('Training the %s -- %s' % (modelClassName, model.type))

        self.classifier = modelClass(**model.params)

        self.classifier = saved or self.classifier.fit(self.X_train[..., self.adhocDependentColsLen:], self.y_train)
        logger.info('Creating a pickle of trained classifier')
        with open(model.picklePath, 'wb') as saved_predictor:
            pickle.dump(self.classifier, saved_predictor, protocol=pickle.HIGHEST_PROTOCOL)

    def modelValidation(self, keyColumn, resultFileName, columnsCount):

        logger.info('Prediction on testing data')
        self.result = self.classifier.predict(self.X_test[..., self.adhocDependentColsLen:])
        self.output = pd.DataFrame(data={"id": self.X_test[..., keyColumn], "sentiment": self.result})
        self.output.to_csv(os.path.join(os.path.dirname(__file__), 'data', \
                                        resultFileName), index=False, quoting=columnsCount)
        logger.info('Writing results to %s' % (resultFileName))
        logger.info('Confusion Matrix:\n')
        cm = confusion_matrix(self.y_test, self.result)
        logger.info(str(cm))
        print(cm)
        logger.info('Classification Metrics:\n%s' % (classification_report(self.y_test, self.result, \
                                                                           target_names=labels)))
        return cm

        # TODO: implement roc curve plot

    def resultGraph(self):

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        positive, negative = 0, 0

        for val in self.result:

            if val == 1:
                positive += 1
            else:
                negative += 1
        logger.info('No of positive responses = %s \nNo of negative responses = %s ' % (positive, negative))
        plt.title('Sentimental Anlaysis')
        plt.pie([negative, positive], labels=labels, startangle=90, shadow=True, \
                explode=(0, 0.1), autopct='%.2f', colors=['red', 'blue'], rotatelabels=True)
        plt.axis('equal')  # make the pie chart circular
        plt.show()

    def sentimentHistogram(self, top_features=40):

        coef = self.classifier.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(self.vectorizer.get_feature_names())
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()

    def moreMetrics(self, model=modelMap.get('svm')):

        global log

        print("=" * 30)
        print(model.modelClass)

        print('****Results****')
        acc = accuracy_score(self.y_test, self.result)
        print("Accuracy: {:.4%}".format(acc))
        ll = log_loss(self.y_test, self.result)
        print("Log Loss: {}".format(ll))

        log_entry = pd.DataFrame([[model.modelClass, acc * 100, ll]], columns=log_cols)
        log = log.append(log_entry)

        print("=" * 30)

        # horizontal bar plot

        import seaborn as sns
        sns.set_color_codes("muted")
        sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

        plt.xlabel('Accuracy %')
        plt.title('Classifier Accuracy')
        plt.show()

        sns.set_color_codes("muted")
        sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

        plt.xlabel('Log Loss')
        plt.title('Classifier Log Loss')
        plt.show()

    def sortedByIDF(self):

        #        self.featureNames = self.vectorizer.get_feature_names()
        #        print("Total number of features: {}\n".format(len(self.featureNames)))
        #        sorted_by_idf = np.argsort(self.vectorizer.idf_)
        #        print("Features with lowest idf:\n{}".format(self.featureNames[sorted_by_idf[:100]]))
        indices = np.argsort(self.vectorizer.idf_)[::-1]
        features = self.vectorizer.get_feature_names()
        top_n = 10
        top_features = [features[-i - 1] for i in indices[:top_n]]
        print(top_features)

    def getCrossValidationScore(self, model=modelMap.get('svm')):

        from sklearn.model_selection import cross_val_score
        import importlib
        modelClassName = model.modelClass
        modelClass = getattr(importlib.import_module(model.libPath), modelClassName)
        logger.info('Training the %s -- %s' % (modelClassName, model.type))
        _classifier = modelClass(**model.params)

        scores = cross_val_score(_classifier, self.X_train[..., self.adhocDependentColsLen:], self.y_train, cv=5)
        print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

    def GridSearch(self, model=modelMap.get('svm'), grid=None):

        import time

        st = time.time

        import importlib
        modelClassName = model.modelClass
        modelClass = getattr(importlib.import_module(model.libPath), modelClassName)
        logger.info('Training the %s -- %s' % (modelClassName, model.type))
        _classifier = modelClass(**model.params)
        self.param_grid = {'C': [10, 100], 'gamma': [0.01, 0.1]}
        self.grid = grid or GridSearchCV(estimator=_classifier, param_grid=self.param_grid, cv=5)
        self.grid.fit(self.X_train[..., self.adhocDependentColsLen:], self.y_train)
        with open(gridPath, 'wb') as saved_grid:
            pickle.dump(self.grid, saved_grid, protocol=pickle.HIGHEST_PROTOCOL)
        print("Test set score: {:.2f}".format(self.grid.score()))
        print("best cross-validation score: {:.2f}".format(self.grid.best_score_))
        print("best parameter: {}".format(self.grid.best_params_))
        print("best estimator: {}".format(self.grid.best_estimator_))
        print("it took {} seconds".format(time.time - st))

    def heatMap(self):

        # extract scores from grid
        score = self.grid.cv_results_['mean_test_score'].shape(-1, 3).T
        # visualize heat map
        heatmap = mglearn.tools.heatmap(
            score, xlabel="C", ylabel="ngram_range", cmap="virdis", fmt="%.3f",
            xticklabels=self.param_grid['C'],
            yticklabels=self.param_grid['gamma'])
        plt.colorbar(heatmap)

    def lDA(self):

        from sklearn.decomposition import LatentDirichletAllocation
        lda = LatentDirichletAllocation(n_topics=10, learning_method="batch",
                                        max_iter=25, random_state=0)
        doc_topics = lda.fit_transform(self.data_features)
        print(doc_topics)
        sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
        feature_names = np.array(self.vectorizer.get_feature_names())
        mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
                                   sorting=sorting, topics_per_chunk=5, n_words=10)


if __name__ == "__main__":
    trainIns = SATrain()
    trainIns.loadData('labeledTrainData.tsv', 1, [0, 2], 3)
    trainIns.dataCleaningNProcessing(1)
    trainIns.documentVectorization(ngram=(1, 2))
    trainIns.splitToTestTrain([0])
    #    trainIns.modelTraining(modelMap.get('random_forest'))
    #                           getSavedPredictor('random_forest'))
    #    trainIns.modelTraining()
    #    trainIns.modelValidation(0,'TestResults25.csv',3)
    #    trainIns.sentimentHistogram(top_features=40)
    #    trainIns.resultGraph()
    #    trainIns.moreMetrics(modelMap.get('random_forest'))
    #    trainIns.moreMetrics()
    #    trainIns.sortedByIDF()
    #    trainIns.getCrossValidationScore(modelMap.get('random_forest'))
    #    trainIns.getCrossValidationScore()
    #    trainIns.GridSearch(grid=saved_grid)
    trainIns.GridSearch()
    trainIns.heatMap()
    trainIns.lDA()

