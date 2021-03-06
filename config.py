#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            config.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/9 15:46    
@Version         1.0 
@Desciption 

'''
from framework.basic.configbase import ConfigBase


class Config(ConfigBase):
    
    def __init__(self, **kwargs):
        super(Config, self).__init__()
        self.train_dataset: str = kwargs['train_dataset']
        self.dev_dataset: str = kwargs['dev_dataset']
        self.test_dataset: str = kwargs['test_dataset']
        
        self.pretrain_model: str = kwargs['pretrain_model']
        self.max_seq_len = kwargs['max_seq_len']
        
        self.hidden_size = kwargs['hidden_size']
        self.label_size = kwargs['label_size']
        
        self.train_batch_size = kwargs['train_batch_size']
        self.test_batch_size = kwargs['test_batch_size']
        
        self.train_times = kwargs['train_times']
        self.epoch = kwargs['epoch']
        
        self.print_iter = kwargs['print_iter']
        self.test_iter = kwargs['test_iter']
        
        self.model_path = kwargs['model_path']
        self.model_name = kwargs['model_name']
        
        self.device = kwargs['device']
        
        # super-params
        self.lr = kwargs['lr']
    
    def from_args(cls):
        pass
