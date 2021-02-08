#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            bce.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/9/14 16:17    
@Version         1.0 
@Desciption 

'''
import torch
import torch.nn as nn


class BCELoss(nn.Module):
    AcceptReduceType = {"mean", "sum", "keep"}
    
    def __init__(self, ignore_index=None, reduce: str = "mean"):
        super(BCELoss, self).__init__()
        if reduce.lower() not in BCELoss.AcceptReduceType:
            raise TypeError("the accept reduce type are %s, but got %s" %
                            (str(BCELoss.AcceptReduceType)), str(reduce))
        
        self.reduce = reduce.lower()
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """

        :param input: [batch_size,1] scores with sigmoid
        :param target: [batch_size, 1]
        :return:
        """
        input = input.reshape(-1)
        target = target.reshape(-1)
        assert input.size(0) == target.size(0), "the target and input must have the same size."
        if self.ignore_index is not None:
            select_pos = target != self.ignore_index
        else:
            select_pos = target < 0
        
        new_input = input[select_pos]
        new_target = target[select_pos]
        return nn.BCELoss(reduction=self.reduce)(new_input, new_target)
