#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            vocabulary.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/8/19 16:31    
@Version         1.0 
@Desciption 

'''

import unittest

from framework.basic.tokenizer import EnglishTokenizer
from framework.basic.vocablary import *


class TestVocab(unittest.TestCase):
    
    def setUp(self) -> None:
        self.tokenizer = EnglishTokenizer()
        self.ori_test_sample1 = ["I like noodles",
                                 "He is a baby boy",
                                 "Who like the pycharm of jet brain."]
        self.test_sample1 = [self.tokenizer.tokenize(item)
                             for item in self.ori_test_sample1]
        
        self.ori_test_sample2 = ["I like noodles"]
        self.test_sample2 = [self.tokenizer.tokenize(item)
                             for item in self.ori_test_sample2]
        
        vocab_file = "../../pretrain/bert-base-uncased/vocab.txt"
        self.vocab = Vocabulary.from_vocab_file(vocab_file)
    
    def testTokenize(self):
        res = self.vocab.tokenize(self.test_sample1, padding=True)
        
        self.assertIsInstance(res, torch.Tensor)
        self.assertEqual(res.dtype, torch.int64)
        self.assertEqual(res.size(), (3, 8))
