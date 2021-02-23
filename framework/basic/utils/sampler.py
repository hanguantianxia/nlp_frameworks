#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            sampler.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/27 17:09    
@Version         1.0 
@Desciption 

'''

import abc
import functools
import math
from collections import Counter, defaultdict
from typing import Union

import numpy as np
import torch

from framework.basic.dataset import Dataset


def _check_ready(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.dataset is not None and self.data_size is not None, \
            "Must get dataset before you use iterator"
        
        return func(self, *args, **kwargs)
    
    return wrapper


class Sampler(torch.utils.data.Sampler):
    mode_candidate = ["idx", "data"]
    """
    the base class of sampler
    """
    
    def __init__(self, mode: Union[str], *args, **kwargs):
        """

        :param mode: return dataloader dataset or dataloader
        :param args:
        :param kwargs:
        """
        mode = mode.lower()
        if mode not in Sampler.mode_candidate:
            raise ValueError("Only accept the mode ['idx', 'data'], but input %s" % mode)
        self.dataset = None
        self.data_size = None
        self._mode = mode
    
    def data(self):
        self._mode = "data"
    
    def idx(self):
        self._mode = "idx"
    
    @property
    def mode(self):
        return self._mode
    
    @abc.abstractmethod
    def init_dataset(self, dataset: Dataset):
        """

        :param dataset:
        :return:
        """
    
    @abc.abstractmethod
    @_check_ready
    def _data_iter(self):
        """
        return the batch generator
        :return:
        """
        for indice_list in self._idx_iter():
            yield self.dataset[indice_list]
    
    @abc.abstractmethod
    @_check_ready
    def _idx_iter(self):
        """
        return the batch id batch generator
        :return:
        """
    
    def __call__(self, batch_size=1):
        raise NotImplementedError
    
    def __iter__(self):
        """
        must
        :return:
        """
        if self._mode == 'idx':
            return self._idx_iter()
        elif self._mode == 'data':
            return self._data_iter()
        else:
            raise ValueError("Only accept the mode ['idx', 'data'], but input %s" % self._mode)
    
    @abc.abstractmethod
    @_check_ready
    def __len__(self):
        """

        :return:
        """


class SequentialSampler(Sampler):
    
    def __init__(self, batch_size=1, drop_last=False, mode="data"):
        super(SequentialSampler, self).__init__(mode)
        
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def init_dataset(self, dataset: Dataset):
        
        self.dataset = dataset
        self.data_size = len(dataset)
        return self
    
    def __len__(self):
        base_times = self.data_size // self.batch_size
        if not self.drop_last:
            base_times += 1
        return base_times
    
    def _idx_iter(self):
        candidate_id_list = range(self.data_size)
        indice = []
        for idx in candidate_id_list:
            indice.append(idx)
            if len(indice) == self.batch_size:
                yield indice
                indice = []
        
        if not self.drop_last and len(indice) != 0:
            yield indice


class RamdomBatchSampler(Sampler):
    
    def __init__(self, batch_size=1, drop_last=False, mode="data"):
        super(RamdomBatchSampler, self).__init__(mode)
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def init_dataset(self, dataset: Dataset):
        self.dataset = dataset
        self.data_size = len(dataset)
        return self
    
    def __len__(self):
        base_times = self.data_size // self.batch_size
        if not self.drop_last:
            base_times += 1
        return base_times
    
    def _idx_iter(self):
        candidate_id_list = np.random.permutation(self.data_size)
        indice = []
        for idx in candidate_id_list:
            indice.append(int(idx))
            
            if len(indice) == self.batch_size:
                yield indice
                indice = []
        
        if not self.drop_last and len(indice) != 0:
            yield indice


class BalanceSampler(Sampler):
    """
    balance sample from by the selected field
    """
    
    def __init__(self, balance_field, batch_size=1, balance_patition=None, drop_last=False, mode="data"):
        """

        :param dataset:
        :param balance_field: only accept the fields have been count
        :param batch_size:
        :param drop_last:
        """
        super(BalanceSampler, self).__init__(mode)
        
        self.balance_field = balance_field
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.balance_patition = balance_patition
    
    def process_balance_filed(self):
        target_field = self.dataset[self.balance_field]
        counter = defaultdict(list)
        freq = Counter(target_field)
        for item_id, item in enumerate(target_field):
            counter[item].append(item_id)
        return counter, freq
    
    def get_class_batch_size_list(self):
        """
        if input [4,4,5]
        :return:
        """
        if self.balance_patition is None or len(self.balance_patition) != self.num_class:
            batch_class_size = self.batch_size // self.num_class
            class_batch_size_list = [batch_class_size] * (self.num_class - 1)
            class_batch_size_list = class_batch_size_list + [self.batch_size - sum(class_batch_size_list)]
        else:
            batch_class_partition = np.array(self.balance_patition).reshape(-1)
            partition = batch_class_partition / batch_class_partition.sum()  # [batch_size]
            class_batch_size_arr = np.array(partition * self.batch_size, dtype=np.int)
            class_batch_size_arr[-1] = self.batch_size - np.sum(class_batch_size_arr[:-1])
            class_batch_size_list = class_batch_size_arr.tolist()
        return class_batch_size_list
    
    def init_dataset(self, dataset: Dataset):
        self.dataset = dataset
        self.data_size = len(dataset)
        class_sample_dict, freq = self.process_balance_filed()
        
        self.freq = freq
        self.num_class = len(self.freq)
        assert self.num_class < self.batch_size, "the batch size must large than class size!"
        
        self.most_common_class = self.freq.most_common(1)[0][0]
        self.most_common_times = self.freq[self.most_common_class]
        
        self.class_batch_size_list = self.get_class_batch_size_list()
        sorted_class_sample_list = sorted(class_sample_dict.items(), key=lambda x: x[0])  # value as sort key
        self.class_sample_list = [val for key, val in sorted_class_sample_list]
        return self
    
    def __len__(self):
        max_len = 0
        for class_size, class_batch_size in zip(self.class_sample_list, self.class_batch_size_list):
            if class_batch_size != 0:
                class_batch_times = math.ceil(len(class_size) / class_batch_size)
                max_len = max(max_len, class_batch_times)
        
        return max_len
    
    def _idx_iter(self):
        # init the random list
        bucket_iterator_list = []
        for sample_list, class_batch_size in zip(self.class_sample_list, self.class_batch_size_list):
            bucket_iterator_list.append(iter(RandomBucket(sample_list, class_batch_size)))
        
        for batch_id in range(len(self)):
            batch_id_list = []
            for class_bucket in bucket_iterator_list:
                batch_id_list.extend(next(class_bucket))
            
            np.random.shuffle(batch_id_list)
            yield batch_id_list


class RandomBucket:
    
    def __init__(self, id_list, batch_size):
        self.id_list = id_list
        self.batch_size = batch_size
    
    def __iter__(self):
        batch = []
        while True:
            if self.batch_size == 0:
                yield []
            else:
                select_ids = np.random.permutation(len(self.id_list))
                
                for select_id in select_ids:
                    batch.append(self.id_list[select_id])
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
