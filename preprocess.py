#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            preprocess.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/9 15:34    
@Version         1.0 
@Desciption 

'''
from framework.basic.dataset import Dataset
from framework.utils.io import *
from transformers import AutoTokenizer
from config import Config
from typing import List, Tuple

fields2methods = {
    "captionID":"keep",
    "gold_label": "keep",
    "pairID": "keep"
}


goldlabel2int = {
    "neutral":0,
    "entailment":1,
    "contradiction":2,
    "-":-1
}

class FieldProcessor:
    
    def __init__(self, config:Config):
        self.config = config
        self.bert_tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model)
    
    
    def search_span(self, bert_input)->List[Tuple[int, int]]:
        """
        search the span of two sentence by sepical token
        [cls] sentence1 [sep] sentence2 [sep]
        the span is like [start, end)
        """
        input_ids = bert_input["input_ids"][0]
        start = 1
        span_list = []
        for pos_idx, token_idx in enumerate(input_ids):
            if token_idx == self.bert_tokenizer.cls_token_id:
                start = pos_idx + 1
            if token_idx == self.bert_tokenizer.sep_token_id:
                span_list.append((start, pos_idx))
                start = pos_idx + 1
        return span_list
        
        
    
    
    
    def preprocess_fields(self, item):
        gold_label = item["gold_label"]
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        
        gold_label_index = goldlabel2int[gold_label]
        
        bert_input = self.bert_tokenizer(
            text=sentence1,
            text_pair=sentence2,
            max_length=self.config.max_seq_len,
            padding='max_length',
            return_tensors="pt"
        )
        
        span = self.search_span(bert_input)
        
        return bert_input, gold_label_index, span
        
        

def prepare_dataset(dataset, config:Config, for_test=False):
    ori_dataset = read_json(dataset)
    dataset = Dataset(ori_dataset)
    if for_test:
        dataset = dataset[:100]
    field_processor = FieldProcessor(config)
    dataset.apply(field_processor.preprocess_fields,
                  new_field_names=['bert_input', 'gold_label_index', "span"],
                  show_process=True)
    
    dataset = dataset.filter(lambda x: x['gold_label_index'] != -1)
    return dataset


if __name__ == '__main__':
    config = Config.from_json("./config/config.json")
    dataset= prepare_dataset(config.train_dataset, config, for_test=True)
    
    
    

