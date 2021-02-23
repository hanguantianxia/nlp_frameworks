#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            dataset.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/29 14:46    
@Version         1.0 
@Desciption 

'''

from framework.basic.dataset import Dataset
from framework.basic.dataset import Instance
from framework.basic.tokenizer import EnglishTokenizer
from framework.utils.io import *

file_path = r"/test/test_data\train-v1.1.jsonl"


def generate_sample_dataset(num_smaples=100):
    """

    :param num_smaples:
    :return:
    """
    data = read_origin_data(file_path, limit=num_smaples)
    data = [Instance(item) for item in data]
    dataset = Dataset(data)
    
    return dataset


def generate_tokned_dataset(num_smaples=100,
                            src_field_names=("context", "question", "answer"),
                            tgt_field_names=("token_context", "token_question", "token_answer")):
    dataset = generate_sample_dataset(num_smaples)
    tokenizer = EnglishTokenizer()
    dataset.apply_fields(tokenizer, src_field_names, tgt_field_names)
    return dataset


def generate_NLI_dataset(num_sample=100):
    def process_label(ins: Instance):
        label = ins['gold_label']
        label = label.lower()
        if label == '-':
            return [0]
        elif label == 'contradiction':
            return [1]
        elif label == 'entailment':
            return [2]
        else:
            return [3]
    
    data = read_origin_data(r"../test_data/snli_1.0_dev.jsonl", limit=num_sample)
    data = [Instance(item) for item in data]
    dataset = Dataset(data)
    dataset.apply(process_label, new_field_names=["label_id"])
    
    return dataset
