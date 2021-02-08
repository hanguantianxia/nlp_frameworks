import copy
from typing import List, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class TfIdf_Model():
    """

    """
    
    def __init__(self,
                 corpus: Union[List[str], List[List[str]]],
                 copy_data=True):
        """

        :param docs: the list of segment documents
        """
        self.vectorizer = CountVectorizer()
        self.model = TfidfTransformer(smooth_idf=True)
        self.copy_data = copy_data
        self.docs = self._process_corpus(corpus)
        self.vectors = self.vectorizer.fit_transform(self.docs)
        
        self.words = None
        self.params = None
        self.params_T = None
        self.sparse = False
    
    def train(self, sparse=False):
        """

        :return:
        """
        self.sparse = sparse
        self.words = self.vectorizer.get_feature_names()
        if sparse:
            self.params = self.model.fit_transform(self.vectors)
            self.params_T = self.params.T.tocsc()
        else:
            self.params = self.model.fit_transform(self.vectors).toarray()
            self.params_T = self.params.T
    
    def sim(self, str1: List[str], str2: List[str]) -> float:
        """
        input two the segment string , compute their similarity by tf-idf
        :param str1: the first string
        :param str2:
        :return:
        """
        
        str1 = self._check_params(str1)
        str2 = self._check_params(str2)
        
        str1_w = self.model.transform(self.vectorizer.transform(str1)).toarray()
        str2_w = self.model.transform(self.vectorizer.transform(str2)).toarray()
        return float(np.dot(str1_w, str2_w.T).flatten())
    
    def topk(self, str1: Union[List[str], List[List[str]]], k, order_by='s') -> Tuple[List[int], List[float]]:
        """
        input segmented string the index of the most similar document for str 1 by
        sort of list and similarity of these docs

        :param str1: list[str] or str the segmented string
        :return: the index of the most similar document for str 1 by sort of list
                 and similarity of them
        """
        if isinstance(str1, str):
            str1 = [str1]
        if isinstance(str1, list):
            if isinstance(str1[0], list):
                str1 = " ".join(str1[0])
        str1_w = self.model.transform(self.vectorizer.transform(str1))
        
        if self.sparse:
            str1_w = str1_w
            similarity = str1_w.dot(self.params_T).toarray().flatten()
        else:
            str1_w = str1_w.toarray()
            similarity = np.dot(str1_w, self.params.T).flatten()
        
        pred_id = np.argpartition(similarity, -k)[-k:]
        
        if order_by == 's':
            pred_id = pred_id[np.argsort(similarity[pred_id])]
        elif order_by == 'i':
            pred_id = np.sort(pred_id)
        else:
            pred_id = pred_id[np.argsort(similarity[pred_id])]
        
        topk_sim = similarity[pred_id]
        
        return pred_id, topk_sim
    
    def _process_corpus(self, corpus):
        assert isinstance(corpus, list)
        if self.copy_data:
            corpus = copy.deepcopy(corpus)
        for item_id, item in enumerate(corpus):
            if isinstance(item, list):
                corpus[item_id] = " ".join(item)
        
        return corpus
    
    def _check_params(self, param):
        if isinstance(param, str):
            param = [param]
        if isinstance(param, list) and len(param) != 1:
            param = [" ".join(param)]
        return param


def test_gensim_bm25():
    corpus = [
        ['来', '问', '几', '个', '问题', '第1', '个', '就', '是', '60', '岁', '60', '岁', '的', '时候', '退休', '是', '时间', '到', '了',
         '一定', '要', '退休', '还是', '觉得', '应该', '差', '不', '多'],
        ['第1', '个', '是', '应该', '第2', '个', '是'],
        ['不', '对', '应该', '就是', '差', '不', '多'],
        ['所以', '是', '应该', '差', '不', '多', '还是', '一定', '要', '退', '60', '岁']]
    
    bm25Model = BM25(corpus)
    
    test_strs = [
        ['所以', '是', '应该', '差', '不', '多', '还是', '一定', '要', '退', '60', '岁'],
        ['所以', '是', '应该', '差', '不', '多', '还是', '一定', '要', '退', '60', '岁', '问题', '第1', '个'],
        ['所以', '是', '应该', '差', '不', '多', '还是', '一定', '要', '退', '60', '岁', '问题', '第1', '个', '来', '问', '几', '个', '问题'],
        ['应该', '差', '不', '多', '一定', '要', '退', '60', '岁'],
        ['差', '不', '多', '一定', '要', '退'],
        ['一定', '要', '差', '不', '多', '退'],
        ['一定', '要', '退'],
        ['一定', '差', '不', '多'],
    ]
    for test_str in test_strs:
        scores = bm25Model.sim(test_str)
        print('测试句子：', test_str)
        for i, j in zip(scores, corpus):
            print('分值：{},原句：{}'.format(i, j))
        print('\n')


if __name__ == '__main__':
    pass
