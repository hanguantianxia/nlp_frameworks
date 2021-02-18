#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            tokenizer.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/18 20:55    
@Version         1.0 
@Desciption 

'''
import abc
import re
from typing import List, Union

import jieba
import nltk
import spacy
from transformers import AutoTokenizer


class BaseTokenizer():
    
    @abc.abstractmethod
    def tokenize(self, input_string: str) -> List[str]:
        """
        this is the main funciton to tokenizer."
        :param input_string: string
        :return: a list of tokens
        """
    
    def __call__(self, input_string) -> List[str]:
        return self.tokenize(input_string)
    
    def count_len(self, input_string):
        return len(self(input_string))

class EnglishTokenizer(BaseTokenizer):
    
    def tokenize(self, input_string: str, lower_case=True) -> List[str]:
        if lower_case:
            input_string = input_string.lower()
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(input_string)]


class SpacyTokenizer(BaseTokenizer):
    
    def __init__(self, corpus="en_core_web_lg"):
        self.nlp = spacy.load(corpus)
    
    def tokenize(self, input_string: Union[str, List[str]], get_lemma=True, lower_case=True, return_parse=False):
        assert isinstance(input_string, str) or isinstance(input_string, list), "Only accept string or a list of string"
        if isinstance(input_string, list):
            input_string = " ".join(input_string)
        
        if lower_case:
            input_string = input_string.lower()
        
        doc = self.nlp(input_string)
        if get_lemma:
            result = [token.lemma_ for token in doc]
        else:
            result = [token.text for token in doc]
        
        if return_parse:
            return result, doc
        else:
            return result
    
    def parse(self, input_string: str):
        return self.nlp(input_string)
    
    def pos_tokenize(self, input_string: str):
        input_string = input_string.lower()
        
        doc = self.nlp(input_string)
        
        pos = [token.pos_ for token in doc]
        result = [token.text for token in doc]
        return result, pos


class JiebaTokenizer(BaseTokenizer):
    def __init__(self, vocab_path=None):
        self.vocab_path = vocab_path
        self.tokenizer = jieba.Tokenizer()
        if self.vocab_path is not None:
            self.tokenizer.load_userdict(vocab_path)
    
    def tokenize(self, input_string: str) -> List[str]:
        """

        """
        
        input_string = re.sub(" ", "", input_string)
        result = self.tokenizer.cut(input_string)
        return list(result)


class BertTokenizer(BaseTokenizer):
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def tokenize(self, input_string: str) -> List[str]:
        return self.tokenizer.tokenize(input_string)
    
    def convert_to_ids(self, input_string: List[str]):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
    
    @staticmethod
    def from_pretrain(filename):
        tokenizer = AutoTokenizer.from_pretrained(filename)
        return BertTokenizer(tokenizer)


if __name__ == '__main__':
    data = "a list of bit."
    
    tokenizer = EnglishTokenizer()
    
    res = tokenizer.tokenize(data)
    tokenizer = AutoTokenizer.from_pretrained(r"E:\Model\BERT\bert-base-uncased")
