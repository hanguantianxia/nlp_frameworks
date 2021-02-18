#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            basic_criterion.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/18 16:00    
@Version         1.0 
@Desciption 

'''
import torch
from abc import abstractmethod
from typing import Dict,Optional, List, Tuple
from framework.basic.batch import Batch
from framework.basic.base_model import BaseModel
class BaseCriterion(BaseModel):
    
    
    @abstractmethod
    def compute_loss(self, logit:Dict, data:Batch)->Optional[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        the abstract method to compute the loss function
        
        :param logit: A dict of the model output
        :param data:  the Batch of input data contains the label or data you need
        :return:
        """
    
    @abstractmethod
    def compute_index(self, logit:Dict, data:Batch)->Dict:
        """
        the abstract method to compute the metrics index
        :param logit:
        :param data:
        :return:
        """
    
    def forward(self,logit:Dict, data:Batch) -> Tuple[Optional[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],Dict]:
        loss = self.compute_loss(logit, data)
        index = self.compute_index(logit, data)
        return loss, index