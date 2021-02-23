#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            span.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/18 17:21    
@Version         1.0 
@Desciption 

'''
from collections import Counter


def compute_span_fpr(span_pred, span_gold):
    span_gold_start, span_gold_end = span_gold
    span_pred_start, span_pred_end = span_pred
    
    gold_toks = list(range(span_gold_start, span_gold_end + 1))
    pred_toks = list(range(span_pred_start, span_pred_end + 1))
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return 0, 0, 0
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, recall, precision
