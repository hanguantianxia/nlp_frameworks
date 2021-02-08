#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            module_template.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/9/8 16:54    
@Version         1.0 
@Desciption 

'''

import torch.nn as nn


class Module(nn.Module):
    
    def __init__(self):
        super(Module, self).__init__()
        self._mode = 'train'
    
    def train(self, mode=True):
        super(Module, self).train(mode)
        self._mode = 'train'
    
    def eval(self):
        super(Module, self).eval()
        self._mode = 'test'
    
    @property
    def mode(self):
        return self._mode
    
    @property
    def device(self):
        for n, p in self.named_parameters():
            return p.device
