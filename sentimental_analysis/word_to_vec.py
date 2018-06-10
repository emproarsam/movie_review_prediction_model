#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:52:16 2018

@author: samarth
"""

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np


class Word2VecGraph(object):
    def __init__(self, documents):
        self.sentences = documents

    def showWordVecs(self, row_index=None, fig_size=(1, 1)):
        # train model
        model = Word2Vec(self.sentences, min_count=1)
        # fit a 2d PCA model to the vectors
        X = model[model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        # create a scatter plot of the projection
        pyplot.figure(figsize=fig_size)
        rows, cols = result.shape
        pyplot.scatter(result[0:row_index or rows, 0], result[0:row_index or rows, 1], s=70, alpha=0.03)
        #        pyplot.xlim(-2, 4)
        #        pyplot.ylim(-8, 8)
        #        pyplot.yticks(np.arange(-2, 4, 0.15))
        #        pyplot.xticks(np.arange(-8, 8, 0.15))
        words = list(model.wv.vocab)
        for i, word in enumerate(words[0:row_index]):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()


if __name__ == "__main__":
    sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    w2vg = Word2VecGraph(sentences)
    w2vg.showWordVecs(row_index=3, fig_size=(30, 4))
