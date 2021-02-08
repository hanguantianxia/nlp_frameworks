#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            bm25.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/7 16:29    
@Version         1.0 
@Desciption 

'''
import numpy as np
from gensim.summarization import bm25


class BM25:
    
    def __init__(self, corpus):
        """

        :param corpus: the list of [docs_id, doc_title, doc_content, segment_doc_content]
        """
        self.docs = [item[-1].split() for item in corpus]  # the segmetn docs_content
        self.ind2docs = {idx: item[0] for idx, item in enumerate(corpus)}  # the segmetn docs_content
        self.docs2ind = {key: value for value, key in self.ind2docs.items()}
        self.model = None
    
    def train(self, sparse=True):
        self.model = bm25.BM25(self.docs)
    
    def topk(self, query, k):
        """

        :param query: it can be str or List[str]
        :param k:
        :return:
        """
        
        self._check_model()
        if isinstance(query, str):
            query = query.split()
        if isinstance(query, list) and len(query) == 1:
            query = query[0].split()
        scores = self.model.get_scores(query)
        scores_arr = np.array(scores).flatten()
        pred_id = np.argpartition(scores_arr, -k)[-k:]
        pred_id = pred_id[np.argsort(scores_arr[pred_id])]
        top_k_sim = scores_arr[pred_id]
        docs_ids = [self.ind2docs.get(idx) for idx in pred_id]
        
        return docs_ids, top_k_sim
    
    def sim(self, query, doc_id):
        self._check_model()
        if isinstance(query, str):
            query = query.split()
        if isinstance(query, list) and len(query) == 1:
            query = query[0].split()
        scores = np.array(self.model.get_scores(query)).flatten()
        idx = self.docs2ind.get(doc_id)
        
        return scores[idx]
    
    def _check_model(self):
        assert self.model is not None, "Please train the model"
