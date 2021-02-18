#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            trainer.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/8/8 17:18    
@Version         1.0 
@Desciption 

'''
import logging
import os
from typing import Dict

import torch
import torch.optim as optim

from framework.basic.git_tool import GitManager
from framework.basic.tester import Tester

class Trainer():
    """
    trainer 需要实现的功能有:
    TODO:
        1. 接受模型进行训练
        2. 模型的保存和训练logger信息的保存
        3. 多种调优方法的实现：
            (1) 梯度累计
            (2) 模型ensemble
            (3) 梯度裁剪
            (4) 梯度优化
        4. 用户的DIY
    
    """
    
    def __init__(self,
                 model,
                 criterion,
                 evaluator,
                 train_loader,
                 dev_loader,
                 optimizer: optim.Optimizer = None,
                 scheduler=None,
                 train_times=1,
                 epoch=5,
                 print_iter=20,
                 test_iter=100,
                 model_path="./model",
                 model_name='',
                 device='cpu',
                 preprocess=None,
                 saved_dict: Dict = None,
                 accumulation_steps=1):
        self.model = model
        self.criterion = criterion
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.EPOCH = epoch
        self.PRINT_ITER = print_iter
        self.TEST_ITER = test_iter
        self.TRAIN_TIMES = train_times
        self.accumulation_steps = accumulation_steps
        
        self.model_path = model_path
        self.best_model_path = os.path.join(self.model_path,
                                            model_name + "_" + type(self.model).__name__ + 'best_model')
        self.checkpoint_path = os.path.join(self.model_path,
                                            model_name + "_" + type(self.model).__name__ + 'checkpoint')
        
        self.device = torch.device(device)
        self.preprocess = preprocess if preprocess is not None else lambda x: x
        
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.logger = self.__init_logger(model_path, model_name)

        
        self.git_manager = GitManager.get_repo(".")
        self.saved_dict = saved_dict
        
    def __init_logger(self,model_path,model_name):
        logger_file_path = os.path.join(model_path, "train_log.log")
        file_handler = logging.FileHandler(filename=logger_file_path)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s - %(name)s %(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
    
        logger = logging.getLogger(model_name)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        return logger
        
        
    def fit(self):
        total_step = 0
        best_indicator = float('inf')
        for train_time in range(self.TRAIN_TIMES):
            self.logger.info("Begin train the %d time!" % train_time)
            
            for epoch in range(self.EPOCH):
                for step, data in enumerate(self.train_loader):
                    total_step += 1
                    data = data.to(self.device)
                    label = label.to(self.device)
                    
                    
                    data = self.preprocess(data)
                    logit = self.model(data)
                    loss,index = self.criterion(logit, data)
                    loss = loss / self.accumulation_steps
                    loss.backward()  # 计算反向回传的梯度
                    
                    if ((step + 1)%self.accumulation_steps) == 0:
                        self.optimizer.step()  # 更新参数，用Adam的方法
                        self.optimizer.zero_grad()  # 清零上一个batch的梯度

                    if total_step % self.PRINT_ITER == 0:
                        msg = "Epoch:[{}/{}] Step:{},Loss {}".format(epoch + 1, self.EPOCH, total_step, loss)
                        self.logger.info(msg)
                    
                    if total_step % self.TEST_ITER == 0:
                        index, metrics = self.evaluator(self.model)
                        if index < best_indicator:
                            msg = "Epoch:[{}epoch/{}] Step:{},Index is better at{}".format(epoch + 1, self.EPOCH,
                                                                                           total_step,
                                                                                           index)
                            self.logger.info(msg)
                            
                            best_indicator = index
                            
                            self.save_model(metrics)
                        else:
                            msg = "Epoch:[{}/{}] Step:{},Index is not better at{}".format(epoch + 1, self.EPOCH,
                                                                                          total_step,
                                                                                          index)
                            self.logger.info(msg)
                        
                        self.save_checkpoint(self.checkpoint_path)
    
    def evalueate(self, data_loader, best_model=True):
        return self.evaluator(data_loader, self.model, self.criterion, self.device)
    
    def save_model(self, metrics):
        save_dict = {
            "state_dict": self.model.state_dict(),
            "metrics": metrics,
            'git_commit_message': self.git_manager._get_commit_mess()
        }
        
        if self.saved_dict is not None:
            save_dict.update(self.saved_dict)
        
        torch.save(save_dict, self.best_model_path + ".pkl")
    
    def save_checkpoint(self, metrics):
        save_dict = {
            "state_dict": self.model.state_dict(),
            "metrics": metrics,
            'git_commit_message': self.git_manager._get_commit_mess(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        if self.saved_dict is not None:
            save_dict.update(self.saved_dict)
        
        torch.save(save_dict, self.best_model_path + "cur_checkpoint.pkl")
    
    @property
    def config(self):
        config = self.__dict__.copy()
        config.pop("model")
        config.pop("criterion")
        config.pop("optimizer")
        config.pop("scheduler")
        
        config.pop("train_loader")
        config.pop("dev_loader")
        config.pop("logger")
        config.pop("git_manager")
        config.pop("preprocess")
        
        return config
        
        
        