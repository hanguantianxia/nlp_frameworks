#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            basic_model.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/18 15:57    
@Version         1.0 
@Desciption 

'''
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Dict
class BaseModel(nn.Module):
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    
    @property
    def device(self):
        return next(iter(self.parameters()))
    
    
    @abstractmethod
    def forward(self,*args, **kwargs) -> Dict:
        """
        
        :param args:
        :param kwargs:
        :return: A dict of result
        """