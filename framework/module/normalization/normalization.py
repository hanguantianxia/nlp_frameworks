#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            normalization.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/9/29 9:33    
@Version         1.0 
@Desciption 

'''

from typing import Union

import torch
import torch.nn as nn


class L12Norm(nn.Module):
    
    def __init__(self, p: Union[int, str] = 2):
        super(L12Norm, self).__init__()
        self.p = int(p)
    
    def forward(self, input):
        if self.p in [1, 2]:
            input_norm = torch.abs(torch.norm(input, p=self.p, dim=-1, keepdim=True)) + 1e-8
            normalized_input = input / input_norm
            return normalized_input
        else:
            return input
