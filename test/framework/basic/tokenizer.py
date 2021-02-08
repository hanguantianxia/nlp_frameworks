#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            tokenizer.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/7/10 16:32    
@Version         1.0 
@Desciption 

'''

from framework.basic.tokenizer import *
import unittest


class TestCase(unittest.TestCase):

    def getInside(self, data):
        for item in data:
            self.assertIsInstance(item, str)

    def setUp(self) -> None:
        self.english_data = "I am a better man."
        self.chinese_data = "我是中华人"
        self.bert_postion = r"..\..\..\pretrain\bert-base-uncased"

    def testJiebaTokenizer(self):
        tokenizer = JiebaTokenizer()

        chi_result = tokenizer.tokenize(self.chinese_data)
        eng_result = tokenizer.tokenize(self.english_data)

        self.assertIsInstance(chi_result, list)
        self.assertIsInstance(eng_result, list)
        self.getInside(chi_result)
        self.getInside(eng_result)

    def testBertTokenizer(self):
        tokenizer = BertTokenizer.from_pretrain(self.bert_postion)

        chi_result = tokenizer.tokenize(self.chinese_data)
        eng_result = tokenizer.tokenize(self.english_data)

        self.assertIsInstance(chi_result, list)
        self.assertIsInstance(eng_result, list)
        self.getInside(chi_result)
        self.getInside(eng_result)

        print(chi_result)
        print(eng_result)
