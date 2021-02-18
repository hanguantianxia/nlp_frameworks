#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            basic_evaluator.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/18 16:00    
@Version         1.0 
@Desciption 

'''
from typing import Tuple,Dict
from abc import abstractmethod
class BaseEvaluator:
    
    @abstractmethod
    def append(self, metrics:Dict):
        """
        
        :param metrics: the metrics dict computed by the criterion instance
        :return:
        """
    
    @abstractmethod
    def reduce(self)->Tuple[float, Dict]:
        """
        
        :return:
        index: the final index to measure the model
        metrics: the metrics of model
        """
    
        
