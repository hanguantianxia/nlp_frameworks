#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            dataloader.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/9 17:14    
@Version         1.0 
@Desciption 

'''
from config import Config
from preprocess import prepare_dataset
from framework.basic.dataset import Dataset
from framework.basic.batch import BatchGenerator
from framework.basic.utils.sampler import *
    

def prepare_dataloader(data_file,
                       sampler=None,
                       wokers=0,
                       batch_size=16,
                       for_test=False):
    """
    prepare one dataset
    
    :param data_file:
    :param sampler:
    :param wokers:
    :param batch_size:
    :param for_test:
    :return:
    """
    dataset = prepare_dataset(data_file, for_test)
    field2methods = {
        "annotator_labels":"keep",
        "captionID": "keep",
        "gold_label": "keep",
        "bert_input": "bert_stack",
        "gold_label_index": "tensor",
        "span":"keep"
    }
    dataloader = BatchGenerator(dataset,
                                field2method=field2methods,
                                batch_sampler=sampler,
                                num_workers=wokers,
                                batch_size=batch_size)
    return dataloader

def prepare_dataloaders(config:Config, for_test=False):
    """
    
    :param config:
    :param for_test:
    :return:
    """
    train_sampler = RamdomBatchSampler(config.train_batch_size)
    train_loader = prepare_dataloader(config.train_dataset, sampler=train_sampler, for_test=for_test)

    train_seq_sampler = SequentialSampler(config.test_batch_size)
    train_seq_loader = prepare_dataloader(config.train_dataset, sampler=train_seq_sampler,for_test=for_test)

    dev_sampler = SequentialSampler(config.test_batch_size)
    dev_loader = prepare_dataloader(config.dev_dataset, sampler=dev_sampler,for_test=for_test)
    
    return train_loader,train_seq_loader, dev_loader

if __name__ == '__main__':
    config = Config.from_json("./config/config.json")
    dataloader = prepare_dataloader(config.train_dataset, config.train_batch_size, for_test=True)
    batch = next(iter(dataloader))
    