#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            batch.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/29 15:31    
@Version         1.0 
@Desciption 

'''

import copy
import unittest
from collections import Counter

from transformers import AutoTokenizer

import framework.basic.utils.sampler as sampler
from framework.basic.batch import BatchGenerator, pad, convert2idx
from framework.basic.vocablary import Vocabulary
from test.framework.test_util import generate_tokned_dataset, generate_sample_dataset, generate_NLI_dataset


class TestCase(unittest.TestCase):
    
    def setUp(self) -> None:
        self.token_dataset = generate_tokned_dataset()
        self.dataset = generate_sample_dataset()
    
    def test_pad_function(self):
        
        vocab = Vocabulary.from_dataset(self.token_dataset, field_names=["token_context"])
        tgt_fields = ['token_context']
        
        padded_dataset = pad(self.token_dataset, ['token_context'], vocab, in_place=False)
        
        item_length = [len(item) for item in padded_dataset[tgt_fields[0]]]
        self.assertEqual(len(set(item_length)), 1)
        self.assertIsNot(padded_dataset, self.token_dataset)
        
        inplace_padded_dataset = pad(self.token_dataset, ['token_context'], vocab, in_place=True)
        
        item_length = [len(item) for item in inplace_padded_dataset[tgt_fields[0]]]
        self.assertEqual(len(set(item_length)), 1)
        self.assertIs(inplace_padded_dataset, self.token_dataset)
    
    def test_convert2idx(self):
        tgt_fields = ['token_context', 'token_question']
        
        vocab = Vocabulary.from_dataset(self.token_dataset, field_names=tgt_fields)
        padded_dataset = pad(self.token_dataset, tgt_fields, vocab)
        origin_dataset = copy.deepcopy(padded_dataset)
        
        idx_dataset = convert2idx(padded_dataset, field_names=tgt_fields, vocab=vocab, )
        
        for tgt_field_name in tgt_fields:
            tgt_field = idx_dataset[tgt_field_name]
            src_field = origin_dataset[tgt_field_name]
            for src_item, tgt_item in zip(src_field, tgt_field):
                for token, idx in zip(src_item, tgt_item):
                    self.assertEqual(vocab.to_token(idx), token)
    
    def test_Batch_Iter(self):
        bert_tokenizer = AutoTokenizer.from_pretrained(r"..\..\..\pretrain\bert-base-uncased")
        
        tgt_fields = ['token_context', 'token_question', 's_idx', 'e_idx', 'id']
        vocab = Vocabulary.from_dataset(self.token_dataset, field_names=tgt_fields)
        
        process_method = ['bert', vocab.tokenize, 'tensor', 'tensor', 'keep']
        batch_size = 3
        
        batch_iter = BatchGenerator(self.token_dataset, vocab, tgt_fields, process_method,
                                    bert_tokenizer=bert_tokenizer, batch_size=batch_size, drop_last=True)
        batch_size_drop = BatchGenerator(self.token_dataset, vocab, tgt_fields, process_method,
                                         bert_tokenizer=bert_tokenizer, batch_size=batch_size, drop_last=False)
        
        total_batches = 0
        for batch in batch_iter:
            self.assertEqual(len(batch), batch_size)
            total_batches += 1
        self.assertEqual(total_batches, len(batch_iter))
        
        total_batches = 0
        for batch in batch_size_drop:
            total_batches += 1
            
            if len(batch) == batch_size:
                self.assertEqual(len(batch), batch_size)
            else:
                self.assertLessEqual(len(batch), batch_size)
        self.assertEqual(total_batches, len(batch_size_drop))


class TestBalanceSampler(unittest.TestCase):
    
    def setUp(self) -> None:
        self.dataset = generate_NLI_dataset()
        vocab = Vocabulary.from_dataset(self.dataset, field_names=['sentence1'])
        self.batch_size = 32
        
        self.sampler = sampler.BalanceSampler(balance_field="label_id", batch_size=self.batch_size,
                                              balance_patition=[0, 0, 0, 1])
        self.generator = BatchGenerator(self.dataset,
                                        vocab,
                                        batch_size=self.batch_size,
                                        field_names=['label_id'],
                                        process_methods=['keep'],
                                        batch_sampler=self.sampler)
    
    def testBalanceSampler(self):
        batch = next(iter(self.generator))
        count_res = Counter(batch.label_id)
        label_size = self.batch_size // len(count_res)
        total_batches = 0
        for key, val in count_res.items():
            self.assertLessEqual(abs(val - label_size), len(count_res))
        
        for _ in self.generator:
            total_batches += 1
        
        self.assertEqual(total_batches, len(self.generator))


if __name__ == '__main__':
    generate_tokned_dataset()
