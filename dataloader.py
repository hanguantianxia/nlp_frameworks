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
import os
from config import Config
from preprocess import prepare_dataset
from framework.basic.dataset import Dataset
from framework.basic.batch import BatchGenerator
from framework.basic.utils.sampler import *
from framework.utils.io import read_pkl

class DataloaderFactory():
    
    def __init__(self, config: Config, for_test:bool=False):
        self.config = config
        self.train_dataset:Union[Dataset,None] = None
        self.dev_dataset:Union[Dataset, None] = None
        self.test_dataset:Union[Dataset,None] = None
        self.for_test = for_test
    
    def prepare_dataloader(self,
                           dataset,
                           sampler=None,
                           wokers=0,
                           batch_size=16):
        """
        prepare one dataset

        :param data_file:
        :param sampler:
        :param wokers:
        :param batch_size:
        :param for_test:
        :return:
        """
        field2methods = {
            "annotator_labels": "keep",
            "captionID": "keep",
            "gold_label": "keep",
            "bert_input": "bert_stack",
            "gold_label_index": "tensor",
            "span": "keep"
        }
        dataloader = BatchGenerator(dataset,
                                    field2method=field2methods,
                                    batch_sampler=sampler,
                                    num_workers=wokers,
                                    batch_size=batch_size)
        return dataloader
    
    def prepare_dataset(self, dataset_file):
        def preprocess_dataset(dataset_file):
            dataset = prepare_dataset(dataset_file, self.config, for_test=self.for_test)
            dataset.save(dataset_file + ".pkl")
            return dataset
            
        def read_pkl_dataset(dataset_file):
            """
            
            :return:
            """
            return read_pkl(dataset_file)
            
        if dataset_file.find(".pkl") != -1:
            dataset = read_pkl_dataset(dataset_file)
        else:
            dataset = preprocess_dataset(dataset_file)
        return dataset
    
    def prepare_datasets(self):
        self.train_dataset = self.prepare_dataset(self.config.train_dataset)
        self.dev_dataset = self.prepare_dataset(self.config.dev_dataset)
        self.test_dataset = self.prepare_dataset(self.config.test_dataset)

    def prepare_dataloaders(self):
        """

        :param config:
        :param for_test:
        :return:
        """
        self.prepare_datasets()
        train_sampler = RamdomBatchSampler(self.config.train_batch_size)
        train_loader = self.prepare_dataloader(self.train_dataset, sampler=train_sampler)
        
        train_seq_sampler = SequentialSampler(self.config.test_batch_size)
        train_seq_loader = self.prepare_dataloader(self.train_dataset, sampler=train_seq_sampler)

        dev_sampler = SequentialSampler(self.config.test_batch_size)
        dev_loader = self.prepare_dataloader(self.dev_dataset, sampler=dev_sampler)

        test_sampler = SequentialSampler(self.config.test_batch_size)
        test_loader = self.prepare_dataloader(self.test_dataset, sampler=test_sampler)

        return train_loader, train_seq_loader, dev_loader,test_loader


if __name__ == '__main__':
    config = Config.from_json("./config/config.json")
    dataloader_factory = DataloaderFactory(config)
    dataloader = dataloader_factory.prepare_dataloaders()[0]
    batch = next(iter(dataloader))
