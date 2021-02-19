#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            evaluator.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/18 17:13    
@Version         1.0 
@Desciption 

'''
from framework.basic.base_evaluator import BaseEvaluator
from typing import Dict,Tuple
from collections import defaultdict
from framework.metrics.fpr import compute_FPRC
class Evaluator(BaseEvaluator):
    
    def __init__(self):
        super(Evaluator, self).__init__()
        self.result_dict = defaultdict(list)
        
        
    def append(self, metrics:Dict):
        for key, value in metrics:
            self.result_dict[key].extend(value)
            
    def reduce(self) ->Tuple[float, Dict]:
        y_true = self.result_dict['gold_label_index']
        y_pred = self.result_dict['select_id_list']
        fprc = compute_FPRC(y_true, y_pred)
        
        f1 = fprc['f1']
        return f1, fprc
        
    