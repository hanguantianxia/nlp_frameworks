#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            tester.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/18 15:40    
@Version         1.0 
@Desciption 

'''
import torch
from tqdm import tqdm

from framework.basic.base_criterion import BaseCriterion
from framework.basic.base_evaluator import BaseEvaluator
from framework.basic.base_model import BaseModel
from framework.basic.batch import BatchGenerator


class Tester:
    
    def __init__(self,
                 dataloader: BatchGenerator,
                 criterion: BaseCriterion,
                 evaluator: BaseEvaluator,
                 preprocess=None,
                 show_process=False
                 ):
        """
        
        :param dataloader:
        :param criterion: compute the loss function
        :param model:
        """
        self.dataloader = dataloader
        self.criterion = criterion
        self.show_process = show_process
        self.preprocess = preprocess if preprocess is not None else lambda x: x
        self.evaluator = evaluator
    
    def set_dataloader(self, new_dataloader):
        self.dataloader = new_dataloader
    
    def set_model(self, new_model):
        self.model = new_model
    
    def eval(self, model: BaseModel):
        iterator = enumerate(self.dataloader)
        if self.show_process:
            iterator = tqdm(iterator)
        
        with torch.no_grad():
            for step, data in iterator:
                data = self.preprocess(data)
                logit = model(data)
                loss, index = self.criterion(logit, data)
                self.evaluator.append(index)
        
        total_index, metrics = self.evaluator.reduce()
        
        return total_index, metrics
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)
