#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            bert_nli_demo.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/9 15:28    
@Version         1.0 
@Desciption 

'''
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from transformers import AutoModel

from config import Config
from dataloader import DataloaderFactory
from framework.basic.base_model import BaseModel
from framework.basic.batch import Batch


class BertNLI(BaseModel):
    
    def __init__(self, config: Config):
        super(BertNLI, self).__init__()
        self.config = config
        self.bert_model = AutoModel.from_pretrained(config.pretrain_model)
        
        self.mlp1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.mlp2 = nn.Linear(config.hidden_size, config.label_size)
    
    def span_pooling(self, bert_output: torch.Tensor, spans: List[Tuple[int, int]]) -> torch.Tensor:
        """
        
        :param bert_output:
        :param spans: List[Tuple[start, end]] the span is [start, end)
        :return:
        """
        bert_pool_list = []
        for batch_id, span in enumerate(spans):
            pool_list = []
            for start, end in span:
                pool_vec = bert_output[batch_id, start:end].mean(dim=0)  # [hidden_size]
                pool_list.append(pool_vec)
            
            pool_vec = torch.cat(pool_list, dim=-1)  # [hidden_size * 2]
            bert_pool_list.append(pool_vec)
        
        bert_pool_vec = torch.stack(bert_pool_list, dim=0)
        return bert_pool_vec
    
    def forward(self, batch: Batch) -> Dict:
        """
        
        :param batch:
        :return:
        """
        bert_input = batch['bert_input']
        span = batch['span']
        
        bert_output = self.bert_model(**bert_input)[0]
        pool_output = self.span_pooling(bert_output, span)
        
        logit = torch.relu(self.mlp1(pool_output))
        logit = self.mlp2(logit)
        
        output_dict = {
            "bert_output": bert_output,
            "bert_pool_output": pool_output,
            "logit": logit
        }
        
        return output_dict


if __name__ == '__main__':
    config = Config.from_json("./config/config.json")
    dataloader_factory = DataloaderFactory(config)
    dataloader = dataloader_factory.prepare_dataloaders()[0]
    
    batch = next(iter(dataloader))
    model = BertNLI(config)
    result = model(batch)
