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
    
class DataloaderFactory():
    
    def __init__(self, config:Config):
        self.config = config

    
    def prepare_dataloader(self,
                           data_file,
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
        dataset = prepare_dataset(data_file,self.config, for_test)
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
    
    def prepare_dataloaders(self,
                            for_test=False):
        """
        
        :param config:
        :param for_test:
        :return:
        """
        train_sampler = RamdomBatchSampler(self.config.train_batch_size)
        train_loader = self.prepare_dataloader(self.config.train_dataset, sampler=train_sampler, for_test=for_test)
    
        train_seq_sampler = SequentialSampler(self.config.test_batch_size)
        train_seq_loader = self.prepare_dataloader(self.config.train_dataset, sampler=train_seq_sampler,for_test=for_test)
    
        dev_sampler = SequentialSampler(self.config.test_batch_size)
        dev_loader = self.prepare_dataloader(self.config.dev_dataset, sampler=dev_sampler,for_test=for_test)
        
        return train_loader,train_seq_loader, dev_loader

if __name__ == '__main__':
    config = Config.from_json("./config/config.json")
    dataloader_factory = DataloaderFactory(config)
    dataloader = dataloader_factory.prepare_dataloaders()[0]

    batch = next(iter(dataloader))
    