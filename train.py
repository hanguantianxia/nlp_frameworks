#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            main.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/10 10:35    
@Version         1.0 
@Desciption 

'''
import torch
from config import Config
from framework.basic.trainer import Trainer
from framework.basic.tester import Tester
from model import BertNLI
from dataloader import DataloaderFactory
from criterian import Criterion
from evaluator import Evaluator
if __name__ == '__main__':
    config = Config.from_json("./config/config.json")
    
    model = BertNLI(config)
    optimizer = torch.optim.AdamW(params=model.parameters())
    criterion = Criterion()
    evaluator = Evaluator()
    
    
    dataloaders_factory = DataloaderFactory(config)
    train_loader, dev_loader, test_loader = dataloaders_factory.prepare_dataloaders()
    
    
    tester = Tester(dev_loader, criterion, evaluator, show_process=True)
    
    
    trainer = Trainer(
        model,
        criterion,
        tester,
        train_loader,
        optimizer,
        train_times=config.train_times,
        print_iter=config.print_iter,
        test_iter=config.test_iter,
        device=config.device,
        model_path=config.model_path,
        model_name=config.model_name
    )
    
    trainer.fit()

