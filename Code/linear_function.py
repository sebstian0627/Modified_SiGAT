#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: h12345jack
@file: logistic_function.py
@time: 2018/12/16
"""

import os
import sys
import re
import time
import json
import pickle
import logging
import math
import random
import argparse
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections import defaultdict

import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from sklearn import linear_model
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import Normalizer

from common import DATASET_NUM_DIC
from fea_extra import FeaExtra

EMBEDDING_SIZE = 20

SINE_MODEL_PATH_DIC = {
    'epinions': './embeddings/sine_epinions_models',
    'slashdot': './embeddings/sine_slashdot_models',
    'bitcoin_alpha': './embeddings/sine_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/sine_bitcoin_otc_models'
}

SIDE_MODEL_PATH_DIC = {
    'epinions': './embeddings/side_epinions_models',
    'slashdot': './embeddings/side_slashdot_models',
    'bitcoin_alpha': './embeddings/side_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/side_bitcoin_otc_models'
}


def read_train_test_data(dataset, k):
    train_X = []
    train_y = []
    with open('final_train_edge.edgelist') as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1)/2)
            train_X.append((i, j))
            train_y.append(flag)
    test_X = []
    test_y = []
    with open('final_test_edge.edgelist') as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1)/2)
            test_X.append((i, j))
            test_y.append(flag)
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)

import matplotlib.pyplot as plt
import seaborn as sns, numpy as np

def common_linear(dataset, k, embeddings, model,epoch):
    train_X, train_y, test_X, test_y  = read_train_test_data(dataset, k)

    train_X1 = []
    test_X1 = []

    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))


    # linear_function = linear_model.LogisticRegression()
    # logistic_function.fit(train_X1, train_y)
    # pred = logistic_function.predict(test_X1)
    # pred_p = logistic_function.predict_proba(test_X1)

    linear_ = linear_model.LinearRegression()
    linear_.fit(train_X1, train_y)
    pred_l = linear_.predict(test_X1)
    print('linear_', metrics.mean_squared_error(test_y, pred_l))
    print(test_y)
    print(pred_l)
    ax1 = sns.distplot(test_y)
    ax1.set(xlabel='x', ylabel='Distribution of X')
    # sns.set(rc={"figure.figsize": (8, 4)}); np.random.seed(0)
    # x = np.random.randn(100)
    ax1 = sns.distplot(pred_l)
    # ax = sns.distplot(x)
    fig = ax1.get_figure()
    fig.savefig("distPlot_pred_"+str(epoch)+ "_original.png") 
    # pos_ratio =  np.sum(test_y) / test_y.shape[0]
    # accuracy =  metrics.accuracy_score(test_y, pred)
    # f1_score0 =  metrics.f1_score(test_y, pred)
    # f1_score1 =  metrics.f1_score(test_y, pred, average='macro')
    # f1_score2 =  metrics.f1_score(test_y, pred, average='micro')

    # auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])
    # print("pos_ratio:", pos_ratio)
    # print('accuracy:', accuracy)
    # print("f1_score:", f1_score0)
    # print("macro f1_score:", f1_score1)
    # print("micro f1_score:", f1_score2)
    # print("auc score:", auc_score)

    return metrics.mean_squared_error(test_y, pred_l)


def read_emb(fpath, dataset):
    dim = 0
    embeddings = 0
    with open(fpath) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                ll = line.split()
                assert len(ll) == 2, 'First line must be 2 numbers'
                dim = int(ll[1])
                embeddings = np.random.rand(DATASET_NUM_DIC[dataset], dim)
            else:
                line_l = line.split()
                node = line_l[0]
                emb = [float(j) for j in line_l[1:]]
                embeddings[int(node)] = np.array(emb)
    return embeddings


def linear_embedding(k=1, dataset='epinions', epoch=10, dirname='sigat'):
    """use sigat embedding to train logistic function
    Returns:
        pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score
    """

    filename = os.path.join('embeddings', dirname, 'embedding-{}-{}-{}_original.npy'.format(dataset, k, epoch))
    file_name1="embedding-bitcoin_alpha-1-200_lasttime.npy"
    file_name2="embedding-bitcoin_alpha-1-200_original.npy"
    embeddings = np.load(file_name1) #toggle file_name1 to file_name2 to obserbve the base model's encoding
    RMSE = common_linear(dataset, k, embeddings, 'sigat',epoch)
    return RMSE

# def logistic_embedding(k=1, dataset='bitcoin_otc', epoch = 10, dirname='sgae'):

#     print(epoch, dataset)
#     fpath = os.path.join(dirname, 'embedding-{}-{}-{}.npy'.format(dataset, k, epoch))
#     embeddings = np.load(fpath)
#     pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings, dirname)
#     return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score


def main():
    dataset = 'bitcoin_alpha'
    rmse = linear_embedding(k=1, dataset=dataset, epoch=200, dirname='sigat')
    # print(rmse)
        


if __name__ == "__main__":
    main()
