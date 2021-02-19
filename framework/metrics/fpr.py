#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            fpr.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/10 10:17    
@Version         1.0 
@Desciption 

'''
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score

def compute_FPRC(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    if len(y_true) != 0:
        f1 = f1_score(y_true, y_pred, average='macro')
        p = precision_score(y_true, y_pred, average='macro')
        r = recall_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        cf = confusion_matrix(y_true, y_pred)
        macro_acc = macro_accuracy(y_true, y_pred)
        message = {"f1": f1,
                   'p': p,
                   "r": r,
                   "cf": cf,
                   "acc": acc,
                   "macro_acc": macro_acc,
                   "y_true": y_true,
                   "y_pred": y_pred,
                   }
    else:
        message = {"f1": 0,
                   'p': 0,
                   "r": 0,
                   "cf": 0,
                   "acc": 0,
                   "macro_acc": 0,
                   "y_true": y_true,
                   "y_pred": y_pred,
                   }
    return message


def macro_accuracy(y_true, y_pred):
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    macro_accuracy = np.mean([conf_mat_norm[i][i] for i in range(conf_mat_norm.shape[0])])
    return macro_accuracy

