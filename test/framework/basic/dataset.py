#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            dataset.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/7 17:52    
@Version         1.0 
@Desciption 

'''
import unittest

from framework.basic.dataset import Dataset, Field
from framework.basic.dataset import Instance
from framework.basic.tokenizer import EnglishTokenizer
from test.framework.test_util import generate_tokned_dataset, generate_sample_dataset

class TestCase(unittest.TestCase):

    def setUp(self) -> None:
        src_field_names = ["context", "question", "answer"]
        tgt_field_names = ["token_context", "token_question", "token_answer"]
        self.token_dataset = generate_tokned_dataset(src_field_names=src_field_names, tgt_field_names=tgt_field_names)
        self.tokenizer = EnglishTokenizer()
        self.dataset = generate_sample_dataset()

    def test_apply_tokenizer(self):
        src_field_names = ["context", "question", "answer"]
        tgt_field_names = ["token_context", "token_question", "token_answer"]
        dataset = generate_tokned_dataset(src_field_names=src_field_names, tgt_field_names=tgt_field_names)

        for tgt_field_name in tgt_field_names:
            field = dataset[tgt_field_name]
            for instnace in field:
                for item in instnace:
                    self.assertIsInstance(item, str)

    def test_combine(self):
        dataset = generate_tokned_dataset()
        combine_style = ['[BOS]', "token_context", '[SEP]', "token_question", '[CLS]']
        tgt_field_name = "combine_data"
        position_field_names = ['s_idx', 'e_idx']
        target_position_fields = ['token_context']
        new_position_field_names = ['combined_s_idx', 'combined_e_idx']

        dataset.combine(combine_style,
                        new_field_names=tgt_field_name,
                        position_field_names=position_field_names,
                        target_position_fields=target_position_fields,
                        new_position_field_names=new_position_field_names)

        for instance in dataset:
            # test new field content
            tgt_token_list = ['[BOS]'] + instance["token_context"] + ['[SEP]'] + instance["token_question"] + ['[CLS]']
            self.assertEqual(tgt_token_list, instance[tgt_field_name])

            # test position change
            tgt_s_pos = instance[position_field_names[0]] + 1
            tgt_e_pos = instance[position_field_names[1]] + 1
            self.assertEqual(tgt_s_pos, instance[new_position_field_names[0]])
            self.assertEqual(tgt_e_pos, instance[new_position_field_names[1]])

    def test_get_item(self):
        ins = self.token_dataset[0]
        slice_dataset_1 = self.token_dataset[0:1]
        slice_dataset_2 = self.token_dataset[0:2]
        list_dataset_1 = self.token_dataset[[0, 1, 2]]
        list_dataset_2 = self.token_dataset[[0]]

        self.assertIsInstance(ins, Instance)
        self.assertIsInstance(slice_dataset_1, Dataset)
        self.assertIsInstance(slice_dataset_2, Dataset)
        self.assertIsInstance(list_dataset_1, Dataset)
        self.assertIsInstance(list_dataset_2, Dataset)

    def testApply(self):
        def function(item: Instance):
            context = item["context"]
            question = item["question"]
            return [self.tokenizer.tokenize(context + question)]

        self.dataset.apply(function, new_field_names=["tokened_ques_context"])
        self.dataset.apply(function, new_field_names=["tokened_ques_context"], show_process=True)

        self.assertIsInstance(self.dataset['tokened_ques_context'], Field)
        for item in self.dataset['tokened_ques_context']:
            self.assertIsInstance(item, list)
            type_list = [isinstance(i, str) for i in item]
            self.assertEqual(all(type_list), True)

    def testFilter(self):

        def judgement(ins: Instance):
            s_idx = ins['s_idx']
            if s_idx < 10:
                return True
            else:
                return False

        new_dataset = self.dataset.filter(judgement)
        for item in new_dataset:
            self.assertLess(item['s_idx'], 10)


if __name__ == '__main__':
    unittest.main()

