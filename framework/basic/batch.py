#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            batch.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/25 12:17    
@Version         1.0 
@Desciption 

'''

__all__ = ['Batch', "BatchGenerator"]

import copy
from typing import Callable
from typing import List, Dict

from torch.utils.data.dataloader import DataLoader

from framework.basic.dataset import *
from framework.basic.utils.sampler import *
from framework.basic.vocablary import Vocabulary
from framework.utils.stat import TimeManager
from .utils.batch import Batch
from .utils.batch_processor import Worker

SINGLE_WORKER = 0


class BatchGenerator():
    process_method = ["bert", "vocab", "keep", "tensor", "bert_stack"]
    
    def __init__(self, dataset: Dataset,
                 vocab=None,
                 field_names=None,
                 process_methods: List[Union[str, None, Callable]] = None,
                 field2method: Dict = None,
                 batch_size=SINGLE_WORKER,
                 num_workers=0,
                 device='cpu',
                 collate_fn=None,
                 max_seq_len=None,
                 bert_tokenizer=None,
                 batch_sampler=None,
                 padding=True,
                 convert_to_id=True,
                 combine=None,
                 type_id=False,
                 drop_last=False,
                 for_test=False,
                 field_rename: Dict = None):
        """
        
        :param dataset:
        :param vocab:
        :param field_names:
        :param process_methods:
        :param field2method:
        :param batch_size:
        :param num_workers:
        :param device:
        :param collate_fn: the function work on batch dataset
        :param max_seq_len:
        :param bert_tokenizer:
        :param batch_sampler:
        :param padding:
        :param convert_to_id:
        :param combine:
        :param type_id:
        :param drop_last:
        :param for_test:
        """
        if field_names is None and process_methods is None:
            if field2method is None:
                raise TypeError("field name, process methods and field2method cannot be None at the same time!")
            else:
                field_names = list(field2method.keys())
                process_methods = list(field2method.values())
        
        assert len(field_names) == len(process_methods), "process method must have the same length as field names"
        _check_process_method(process_methods)
        if vocab is not None:
            assert isinstance(vocab, Vocabulary), "Vocab must be the Vocabulary instance."
        assert isinstance(dataset, Dataset), "dataset must be the Dataset instance."
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.convert_to_id = convert_to_id
        self.combine = combine
        self.type_id = type_id
        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = batch_size
        
        self.field_names = field_names
        self.process_methods = process_methods
        self.field_rename = field_rename
        
        self.field2method = {k: v for k, v in zip(self.field_names, self.process_methods)}
        if field2method is not None:
            self.field2method.update(field2method)
        
        self.sampler = SequentialSampler(batch_size, drop_last) \
            if batch_sampler is None else batch_sampler
        
        self.sampler.idx()
        self.drop_last = drop_last
        self.bert_tokenizer = bert_tokenizer
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        
        self.for_test = for_test
        self.clock = TimeManager()
        
        self.device = torch.device(device)
        
        self.dataset_sampler = self.sampler.init_dataset(self.dataset)
        self.preprocess_worker = Worker(field2method=self.field2method,
                                        bert_tokenizer=bert_tokenizer,
                                        vocab=vocab,
                                        max_seq_len=max_seq_len,
                                        collate_fn=collate_fn,
                                        multi_process=False,
                                        field_rename=self.field_rename
                                        )
        self.worker = CollateWrapper(self.preprocess_worker)
        
        self.torch_dataloader = DataLoader(dataset=self.dataset,
                                           batch_sampler=self.dataset_sampler,
                                           collate_fn=self.worker,
                                           num_workers=self.num_workers)
    
    def __iter__(self):
        for batch in self.torch_dataloader:
            batch.to(self.device)
            yield batch
    
    def __len__(self):
        return len(self.dataset_sampler)


class CollateWrapper:
    
    def __init__(self, worker: Worker):
        self.worker = worker
    
    def __call__(self, data_list: List[Instance]):
        minidataset = Dataset(data_list)
        return self.worker(0, minidataset)


def _check_process_method(process_methods):
    new_process = []
    for item in process_methods:
        if isinstance(item, str):
            assert item.lower() in BatchGenerator.process_method, \
                "The must process only support {} or function(item)".format(BatchGenerator.process_method)
            item = item.lower()
        elif callable(item):
            pass
        else:
            raise TypeError("The must process only support {} or function(item)".format(BatchGenerator.process_method))
        
        new_process.append(item)
    return new_process


def pad(dataset: Dataset, field_names, vocab: Vocabulary, max_len=None, in_place=True) -> Dataset:
    if isinstance(field_names, list):
        for field_name in field_names:
            assert isinstance(field_name, str)
    elif isinstance(field_names, str):
        field_names = [field_names]
    else:
        raise TypeError("Only support the list of string or string.")
    
    if not in_place:
        dataset = copy.deepcopy(dataset)
    for field_name in field_names:
        field = dataset[field_name]
        padder = field.padder(vocab.pad_token)
        padded_field = padder(field, max_len=max_len, in_place=True)
        dataset[field_name] = padded_field
    
    return dataset


def convert2idx(dataset: Dataset,
                field_names,
                vocab: Vocabulary,
                in_place=True,
                bert_tokenizer=None) -> Dataset:
    if not in_place:
        dataset = copy.deepcopy(dataset)
    
    for field_name in field_names:
        field = dataset[field_name]
        for idx, item in enumerate(field):
            id_item = [vocab.to_index(token) for token in item]
            field[idx] = id_item
        dataset[field_name] = field
    return dataset
