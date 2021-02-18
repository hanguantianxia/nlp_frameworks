#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            criterian.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/10 9:29    
@Version         1.0 
@Desciption 

'''

from typing import Dict

import torch.nn as nn

from framework.basic.batch import Batch


class Criterian:
    
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()
    
    def compute_loss(self, model_output: Dict, batch: Batch):
        """
        
        :param model_output:
        :param batch:
        :return:
        """
        label = batch['gold_label_index']
        logit = model_output['logit']
        loss = self.loss(logit, label)
        return loss
    
    def __call__(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)
