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
import os

import torch
import torch.nn as nn
import torch.optim as optim

from framework.utils.io import write_pkl


class Trainer():
    
    def __init__(self, model,
                 criterion,
                 evaluate,
                 train_loader,
                 dev_loader,
                 optimizer=optim.Adam,
                 epoch=5,
                 print_iter=20,
                 test_iter=100,
                 model_path="./model",
                 max_check_point=5,
                 device='cpu',
                 preprocess=None):
        self.model = model
        self.criterion = criterion
        self.evaluate = evaluate
        self.optimizer = optimizer(model.parameters())
        
        self.EPOCH = epoch
        self.PRINT_ITER = print_iter
        self.TEST_ITER = test_iter
        self.total_step = 0
        
        self.model_path = model_path
        self.best_model_path = os.path.join(self.model_path, type(self.model).__name__ + 'best_model')
        self.current_model_path = os.path.join(self.model_path, type(self.model).__name__ + 'current_model')
        
        self.device = torch.device(device)
        self.preprocess = preprocess if preprocess is not None else lambda x: x
        
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        
        self.model.to(self.device)
        self.criterion.to(self.device)
    
    def fit(self):
        total_step = 0
        best_indicator = float('inf')
        for epoch in range(self.EPOCH):
            for step, (data, label) in enumerate(self.train_loader):
                total_step += 1
                data = data.to(self.device)
                label = label.to(self.device)
                
                self.optimizer.zero_grad()  # 清零上一个batch的梯度
                
                data = self.preprocess(data)
                logit = self.model(data)
                loss = self.criterion(logit, label)
                
                loss.backward()  # 计算反向回传的梯度
                self.optimizer.step()  # 更新参数，用Adam的方法
                
                if total_step % self.PRINT_ITER == 0:
                    print("Epoch:[{}/{}] Step:{},Loss {}".format(epoch + 1, self.EPOCH, total_step, loss))
                if total_step % self.TEST_ITER == 0:
                    index, _ = self.evaluate(self.dev_loader, self.model, self.criterion, self.device)
                    if index < best_indicator:
                        print("Epoch:[{}/{}] Step:{},Index is better at{}".format(epoch + 1, self.EPOCH, total_step,
                                                                                  index))
                        best_indicator = index
                        self.save_model(self.best_model_path)
                    else:
                        print("Epoch:[{}/{}] Step:{},Index is not better at{}".format(epoch + 1, self.EPOCH, total_step,
                                                                                      index))
                    self.save_model(self.current_model_path)
        
        return self.model
    
    def evalueate(self, data_loader, best_model=True):
        return self.evaluate(data_loader, self.model, self.criterion, self.device)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path + ".model")
        torch.save(self.optimizer.state_dict(), path + ".optim")
    
    def return_best_model(self):
        return self.model.load_state_dict(self.best_model_path + '.model')
    
    def save_check_point(self):
        pass


class TrainerComplex():
    
    def __init__(self, model,
                 criterion,
                 evaluate,
                 train_loader,
                 dev_loader,
                 optimizer=optim.Adam,
                 epoch=5,
                 print_iter=20,
                 test_iter=100,
                 model_path="./model",
                 max_check_point=5,
                 device='cpu',
                 preprocess=None,
                 scheduler=None,
                 clip_value=5
                 ):
        
        self.model = model
        self.criterion = criterion
        self.evaluate = evaluate
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.epoch = epoch
        self.PRINT_ITER = print_iter
        self.TEST_ITER = test_iter
        self.total_step = 0
        self.clip_value = clip_value
        
        self.model_path = model_path
        self.best_model_path = os.path.join(self.model_path, type(self.model).__name__ + 'best_model')
        self.current_model_path = os.path.join(self.model_path, type(self.model).__name__ + 'current_model')
        
        self.device = torch.device(device)
        self.preprocess = preprocess if preprocess is not None else lambda x: x
        
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        
        self.model.to(self.device)
        self.criterion.to(self.device)
    
    def fit(self):
        total_step = 0
        best_indicator = -float('inf')
        
        for epoch in range(self.epoch):
            for step, batch in enumerate(self.train_loader):
                total_step += 1
                batch.to(self.device)
                
                loss, messages = self.model(batch)
                
                self.optimizer.zero_grad()  # 清零上一个batch的梯度
                loss.backward()  # 计算反向回传的梯度
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)
                self.optimizer.step()  # 更新参数，用Adam的方法
                self.scheduler.step()
                
                if total_step % self.PRINT_ITER == 0:
                    print("Epoch:[{}/{}] Step:{},Loss {}".format(epoch + 1, epoch, total_step, loss))
                if total_step % self.TEST_ITER == 0:
                    index, metrics = self.evaluate(self.dev_loader, self.model)
                    write_pkl(os.path.join(self.model_path, 'message.pkl'), metrics)
                    if index < best_indicator:
                        print("Epoch:[{}/{}] Step:{},Index is better at{}".format(epoch + 1, epoch, total_step, index))
                        best_indicator = index
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.model_path, "bestmodel.pt"))
                    else:
                        print("Epoch:[{}/{}] Step:{},Index is not better at{}".format(epoch + 1, epoch, total_step,
                                                                                      index))
    
    def save(self):
        pass
    
    def model(self):
        pass
