#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            batch.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/11/18 16:15    
@Version         1.0 
@Desciption 

'''
from typing import Union

import torch
import transformers

from ..dataset import Dataset


class Batch():
    type_list = ['numpy', 'torch']
    
    def __init__(self, batch: Union[Dataset, None] = None, type="numpy"):
        """

        :param batch:
        :param type:
        """
        assert type.lower() in Batch.type_list, "Now we only support the {}.".format(Batch.type_list)
        self._batch_data = {"_dataset": batch}
        self._dataset = self._batch_data['_dataset']
        self._type = type
    
    def to(self, device: torch.device):
        for key, val in self._batch_data.items():
            self[key] = self._to(val, device)
    
    def _to(self, obj, device: torch.device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, list):
            for item_id, item in enumerate(obj):
                obj[item_id] = self._to(item, device)
            return obj
        elif isinstance(obj, dict):
            for key, val in obj.items():
                obj[key] = self._to(val, device)
            return obj
        elif isinstance(obj, transformers.tokenization_utils_base.BatchEncoding):
            for key, val in obj.items():
                obj[key] = self._to(val, device)
            return obj
        else:
            return obj
    
    def __setitem__(self, key, value):
        assert isinstance(key, str), "Only accept key data is dataset"
        self._batch_data[key] = value
        setattr(self, key, value)
    
    def __getitem__(self, item):
        return self._batch_data[item]
    
    def __len__(self):
        return len(self._dataset)
    
    def __repr__(self):
        return str(self._batch_data)
    
    def keys(self):
        return self._batch_data.keys()
    
    def __getattr__(self, item):
        
        return self.__dict__.get(item, None)
    
    def to_dict(self):
        return self._batch_data
    
    @classmethod
    def from_dict(cls, batch_dict: dict):
        batch = cls()
        for key, val in batch_dict.items():
            batch[key] = val
        return batch
    
    def __getstate__(self):
        return self.to_dict()
    
    def __setstate__(self, state):
        for k, v in state.items():
            self.k = v
        self._batch_data = state
