#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            field.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/29 15:26    
@Version         1.0 
@Desciption 

'''

import unittest

from framework.basic.field import DefaultPadder
from test.framework.test_util import generate_tokned_dataset


class TestCase(unittest.TestCase):
    def test_field(self):
        dataset = generate_tokned_dataset()
        select_ids = [0]
        c = dataset[[0]]

        for field_name, field in c.items():
            self.assertEqual(len(field), 1)

    def test_Default_Padder_inplace(self):
        dataset = generate_tokned_dataset()
        padder = DefaultPadder()
        src_field = dataset['token_context']
        max_len = max([len(item) for item in src_field])

        padded_field = padder(src_field, in_place=True)
        self.assertEqual(padded_field, src_field)

        for item in padded_field:
            self.assertEqual(len(item), max_len)

    def test_Default_Padder(self):
        dataset = generate_tokned_dataset()
        padder = DefaultPadder()
        src_field = dataset['token_context']
        max_len = max([len(item) for item in src_field])

        padded_field = padder(src_field, in_place=False)
        self.assertNotEqual(padded_field, src_field)

        for item in padded_field:
            self.assertEqual(len(item), max_len)
