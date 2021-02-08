#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            batch_processor.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/11/18 16:18    
@Version         1.0 
@Desciption 

'''
import os
from abc import abstractmethod
from multiprocessing import Queue
from typing import List, Dict, Callable, Union

import torch

from .batch import Batch
from ..dataset import Dataset
from ..vocablary import Vocabulary


class BaseWorker:
    
    def __init__(self, multi_process=False):
        self._multi_process = multi_process
    
    @abstractmethod
    def single_process(self, batch_id, mini_dataset: Dataset):
        """

        :param batch_id:
        :param mini_dataset:
        :return:
        """
    
    def __call__(self, *args, **kwargs):
        if self._multi_process:
            self.multi_process(*args, **kwargs)
        else:
            return self.single_process(*args, **kwargs)
    
    def multi_process(self, q_in: Queue, q_out: Queue):
        while True:
            batch_id, batch_data = q_in.get()
            res = self.single_process(batch_id, batch_data)
            print("BaseWorker {} process  batch_id {}".format(os.getpid(), batch_id))
            q_out.put([batch_id, res], block=True)
            print("BaseWorker {} process finish batch_id {}".format(os.getpid(), batch_id))


class Worker(BaseWorker):
    
    def __init__(self,
                 field_names=None,
                 process_methods: List[Union[str, None, Callable]] = None,
                 field2method: Dict = None,
                 bert_tokenizer=None,
                 vocab: Vocabulary = None,
                 max_seq_len: int = None,
                 multi_process=False,
                 collate_fn: Union[None, Callable] = None,
                 field_rename: Union[None, Dict] = None):
        """

        :param field_names:
        :param process_methods:
        :param field2method:
        :param bert_tokenizer:
        :param vocab:
        :param max_seq_len:
        :param multi_process:
        :param collate_fn: combine function to get all process
        :param field_rename: rename the field name after the batch processingf
        """
        
        super(Worker, self).__init__(multi_process)
        if field_names is None and process_methods is None:
            if field2method is None:
                raise TypeError("field name, process methods and field2method cannot be None at the same time!")
            else:
                field_names = list(field2method.keys())
                process_methods = list(field2method.values())
        self.field_names = field_names
        self.process_methods = process_methods
        self.field2method = field2method
        self.bert_tokenizer = bert_tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.collate_fn = collate_fn if collate_fn is not None else lambda x: x
        self.field_rename = field_rename if field_rename is not None else dict()
    
    def single_process(self, batch_id, mini_dataset: Dataset):
        """

        :param mini_dataset:
        :return:
        """
        mini_batch = Batch(mini_dataset)
        
        for field_name, process_method in zip(self.field_names, self.process_methods):
            field = mini_dataset[field_name]
            if isinstance(process_method, str):
                process_method = process_method.lower()
                if process_method == 'bert':
                    assert self.bert_tokenizer is not None, \
                        "Must input the bert tokenizer of transformers to the BatchGenerator"
                    process_result = self.bert_tokenizer(field.data, padding=True, is_pretokenized=True,
                                                         return_tensors='pt', add_special_tokens=False,
                                                         max_length=self.max_seq_len)
                elif process_method == 'vocab':
                    process_result = self.vocab.tokenize(field.data, max_length=self.max_seq_len)
                elif process_method == 'tensor':
                    process_result = torch.tensor(field.data)
                else:
                    process_result = field.data
            elif callable(process_method):
                process_result = process_method(field.data)
            else:
                process_result = field.data
            if field_name in self.field_rename:
                field_name = self.field_rename[field_name]
            mini_batch[field_name] = process_result
        
        mini_batch = self.collate_fn(mini_batch)
        
        mini_batch.to('cpu')
        return mini_batch
