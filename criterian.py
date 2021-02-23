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
from typing import Dict, Union, List

import torch
import torch.nn as nn

from framework.basic.base_criterion import BaseCriterion
from framework.basic.batch import Batch


class Criterion(BaseCriterion):
    
    def __init__(self):
        super(Criterion, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def compute_loss(self, logit: Dict, data: Batch) -> Union[
        torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        
        :param model_output:
        :param batch:
        :return:
        """
        label = data['gold_label_index']
        logit = logit['logit']
        loss = self.loss(logit, label)
        return loss
    
    def get_index(self, logit: Dict, data: Batch) -> Dict:
        """
        
        :param logit:
        :param data:
        :return:
        """
        logit = logit['logit']
        select_id = torch.argmax(logit, dim=-1)  # [batch_size]
        select_id_list = select_id.tolist()
        
        gold_label_index = data['gold_label_index']
        gold_label_index_list = gold_label_index.tolist()
        
        index = {
            "select_id_list": select_id_list,
            "gold_label_index": gold_label_index_list
        }
        return index
