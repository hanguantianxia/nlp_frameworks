#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            normalization.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/9/29 9:41    
@Version         1.0 
@Desciption 

'''
import unittest
from framework.module.normalization.normalization import *


class TestL12Norm(unittest.TestCase):

    def setUp(self) -> None:
        self.data = torch.rand(4, 4, 2)

    def testNone(self):
        norm = L12Norm(p=0)
        res = norm(self.data)

        judge = torch.abs(res - self.data) < 1e-5
        self.assertTrue(torch.all(judge))

    def testL1(self):
        norm = L12Norm(p=1)
        res = norm(self.data)

        res_l1_norm = torch.norm(res, p=1, dim=-1, keepdim=True)
        judge = torch.abs(res_l1_norm - 1) < 1e-5
        self.assertTrue(torch.all(judge))

    def testL2(self):
        norm = L12Norm(p=2)
        res = norm(self.data)

        res_l1_norm = torch.norm(res, p=2, dim=-1, keepdim=True)
        judge = torch.abs(res_l1_norm - 1) < 1e-5
        self.assertTrue(torch.all(judge))
