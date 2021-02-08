#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            vocablary.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/18 16:40    
@Version         1.0 
@Desciption 

'''
import copy
import functools
from collections import Counter
from collections import Iterable
from typing import Union

import numpy as np
import torch

from framework.basic.dataset import Dataset, Instance
from framework.utils.io import *

Default_Remain_Token_Num = 100
Default_Pad_Token = "[PAD]"
Default_Unk_Token = "[UNK]"
Default_Special_Token_List = ["<bos>", "<eos>", "<sep>", "<cls>"]


class Vocabulary_Error(Exception):
    pass


def _check_vocab_ready(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.build_vocab_flag:
            return func(self, *args, **kwargs)
        else:
            info_msg = "The vocaulary is not build." \
                       "Please implement vocabulary.build_vocab()."
            # Todo:
            #   1. Log
            raise Vocabulary_Error(info_msg)
    
    return wrapper


def _check_vocab_block(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.block:
            return func(self, *args, **kwargs)
        else:
            info_msg = "The vocaulary is  block. Cannot insert any word into vocab. " \
                       "You can use unlock method to reset."
            # Todo:
            #   1.Log
    
    return wrapper


class Vocabulary:
    """
    The class
    1. record the vocabulary of dataset
    2. convert index to token
    3.
    """
    return_type = ['np', 'pt', 'list']
    
    def __init__(self, max_size=None, min_freq=None,
                 remain_num=None, padding=Default_Pad_Token,
                 unknown=Default_Unk_Token,
                 special_token_list=Default_Special_Token_List):
        
        self.max_size = max_size
        self.min_freq = min_freq
        self.remain_vocab = remain_num
        
        self._idx2word = None
        self._word2idx = None
        self.freq_vocab = Counter()
        self.remain_idx_list = []
        self.remain_token_list = []
        
        self.pad_token = padding
        self.unk_token = unknown
        self.pad_token_id = None
        self.unk_token_id = None
        self.special_token_list = special_token_list
        
        self.build_vocab_flag = False
        self.block = False
    
    @_check_vocab_block
    def add_word_list(self, words):
        assert isinstance(words, list), ""
        for word in words:
            self.add(word)
    
    @_check_vocab_block
    def add_word(self, word):
        """
        add a word into vocabulary
        :param word:
        :return:
        """
        self.build_vocab_flag = False
        self.add(word)
    
    @_check_vocab_block
    def add(self, word):
        assert isinstance(word, str), "the vocabulary must be a string"
        self.build_vocab_flag = False
        self.freq_vocab[word] += 1
        return self
    
    def lock(self):
        self.block = True
    
    @_check_vocab_block
    def unlock(self):
        self.block = False
        self.clear()
    
    def build_vocab(self, ignore_sp_token=False):
        """
        build the vocabulary by max_len, min_freq, freq_vocab
        change the word2idx, idx2word
        :return:
        """
        
        self._word2idx = {}
        # add pad and special token
        if not ignore_sp_token:
            if self.pad_token is not None:
                self._word2idx[self.pad_token] = len(self._word2idx)
                self.pad_token_id = self._word2idx[self.pad_token]
                self._word2idx[self.unk_token] = len(self._word2idx)
                self.unk_token_id = self._word2idx[self.unk_token]
            
            for special_token in self.special_token_list:
                self._word2idx[special_token] = len(self._word2idx)
        
        max_size = min(self.max_size, len(self.freq_vocab)) if self.max_size else None
        words = self.freq_vocab.most_common(max_size)
        
        if self.min_freq is not None:
            words = [item for item in words if item[1] >= self.min_freq]
        if len(self._word2idx) != 0:
            words = [item for item in words if item[0] not in self._word2idx]
        
        words = [item[0] for item in words]
        
        # remain only once
        if self.remain_vocab is not None and self.build_vocab_flag:
            while len(self._word2idx) <= self.remain_vocab:
                remain_token = "remain_%d" % len(self)
                self._word2idx[remain_token] = len(self)
                self.remain_idx_list.append(len(self))
                self.remain_token_list.append(remain_token)
        
        for word in words:
            self._add_word2idx(word)
        self._build_reverse_vocab()
        self.build_vocab_flag = True
        
        if (self.unk_token is not None) and (self.unk_token != self.pad_token):
            self.unk_token_id = self._word2idx[self.unk_token]
            self.pad_token_id = self._word2idx[self.pad_token]
        
        return self
    
    def _build_reverse_vocab(self):
        self._idx2word = {v: k for k, v in self._word2idx.items()}
    
    def _add_word2idx(self, word):
        if word not in self._word2idx:
            self._word2idx[word] = len(self._word2idx)
    
    @_check_vocab_ready
    def to_index(self, word):
        return self._word2idx.get(word, self.unk_token_id)
    
    @_check_vocab_ready
    def to_token(self, idx):
        return self._idx2word.get(idx, self.unk_token)
    
    @_check_vocab_ready
    def convert_ids_to_tokens(self, ids):
        """

        :param ids:
        :return:
        """
        assert isinstance(ids, Iterable), "Input must be a iterable sequence of string."
        return [self.to_token(index) for index in ids]
    
    def convert_tokens_to_ids(self, tokens: List[str], return_type: str = 'np'):
        """

        :param tokens:
        :return:
        """
        assert isinstance(tokens, Iterable), "Input must be a iterable sequence of string."
        assert return_type.lower() in Vocabulary.return_type, "the return type must be in {}".format(
            Vocabulary.return_type)
        
        ids_data = [self.to_index(token) for token in tokens]
        if return_type.lower() == 'np':
            ids_data = np.array(ids_data)
        elif return_type.lower() == 'pt':
            ids_data = torch.tensor(ids_data)
        elif return_type.lower() == 'list':
            pass
        
        return ids_data
    
    def clear(self):
        self.max_size = None
        self.min_freq = None
        self.remain_vocab = None
        
        self._idx2word = None
        self._word2idx = None
        self.freq_vocab = Counter()
        self.remain_idx_list = []
        self.remain_token_list = []
        
        self.pad_token = Default_Pad_Token
        self.unk_token = Default_Unk_Token
        self.pad_token_id = None
        self.unk_token_id = None
        self.special_token_list = Default_Special_Token_List
        
        self.build_vocab_flag = False
        self.remain_token_num = Default_Remain_Token_Num
        return self
    
    @_check_vocab_ready
    def __getitem__(self, item):
        if isinstance(item, str):
            return self._word2idx.get(item, self.unk_token_id)
        elif isinstance(item, int):
            return self._idx2word.get(item, self.unk_token)
        else:
            # todo : log Warning
            return self._idx2word[int(item)]
    
    @_check_vocab_ready
    def __len__(self):
        return len(self._word2idx)
    
    @_check_vocab_ready
    def __contains__(self, item):
        return item in self._word2idx
    
    @_check_vocab_ready
    def tokenize(self, text: Union[List[str], List[List[str]]],
                 padding: Union[bool, str] = True,
                 max_length: Union[int, None] = None,
                 is_pretokenized=False,
                 return_tensors: str = 'pt'):
        assert isinstance(text, list), "Only accept the list of string or a list of a list of vocab."
        assert len(text) > 0, "Don't accept empty list"
        if padding == False and return_tensors == 'list':
            raise TypeError("When return pytorch tensor,  padding must equal to True.")
        
        def convert_list_of_str(text):
            return self.convert_tokens_to_ids(text, return_type='list')
        
        def pad(text):
            batch_max_len = max([len(item) for item in text])
            text = copy.deepcopy(text)
            if max_length is not None:
                max_len = max(max_length, batch_max_len)
            else:
                max_len = batch_max_len
            for item in text:
                ori_len = len(item)
                pad_list = [self.pad_token] * (max_len - ori_len)
                item.extend(pad_list)
            return text
        
        def convert_list_list_str(text):
            if padding:
                text = pad(text)
            new_text = [convert_list_of_str(item) for item in text]
            return new_text
        
        sample = text[0]
        if isinstance(sample, str):
            ids = convert_list_of_str(text)
        elif isinstance(sample, list):
            ids = convert_list_list_str(text)
        else:
            raise TypeError("Only accept the list of string or a list of a list of string.")
        
        if return_tensors == 'pt':
            return torch.tensor(ids, dtype=torch.int64)
        elif return_tensors == 'np':
            return np.array(ids, dtype=np.int64)
        else:
            return ids
    
    @_check_vocab_block
    def from_datasets(self, *datasets, field_names: List[str], max_size=None, remain_num=None, min_freq=None):
        """

        :param datasets:
        :param max_size:
        :param remain_num:
        :param min_freq:
        :return:
        """
        self.max_size = max_size,
        self.min_freq = min_freq,
        self.remain_token_num = remain_num
        
        if isinstance(field_names, str):
            field_names = [field_names]
        elif not isinstance(field_names, list):
            raise TypeError("invalid argument field name: {}".format(field_names))
        
        def get_vocab(ins: Instance):
            for field_name in field_names:
                assert field_name in ins.keys(), "the field name %s not in the dataset" % field_name
            
            for field_name in field_names:
                field = ins[field_name]
                if isinstance(field, list):
                    for word in field:
                        assert not isinstance(word, str), "the field should be a list of str not %s" % type(word)
                        self.add_word(word)
                if isinstance(field, str):
                    # assume the self should have been tokenized.
                    self.add_word(field)
        
        for idx, dataset in enumerate(datasets):
            if not isinstance(dataset, dataset.Dataset):
                raise TypeError("Only accept Dataset Type.")
            dataset.apply(get_vocab)
        return self
    
    @staticmethod
    def from_dataset(*datasets, field_names: List[str], max_size=None, remain_num=None, min_freq=None):
        """

        :param dataset: dataset list to address the train dataset and dev set
        :param field_names:
        :return:
        """
        vocab = Vocabulary(max_size=max_size,
                           min_freq=min_freq,
                           remain_num=remain_num)
        if isinstance(field_names, str):
            field_names = [field_names]
        elif not isinstance(field_names, list):
            raise TypeError("invalid argument field name: {}".format(field_names))
        
        def get_vocab(ins: Instance):
            for field_name in field_names:
                assert field_name in ins.keys(), "the field name %s not in the dataset" % field_name
            
            for field_name in field_names:
                field = ins[field_name]
                if isinstance(field, list):
                    for word in field:
                        assert isinstance(word, str), "the field should be a list of str not %s" % type(word)
                        vocab.add_word(word)
                if isinstance(field, str):
                    # assume the vocab should have been tokenized.
                    vocab.add_word(field)
        
        for idx, dataset in enumerate(datasets):
            if not isinstance(dataset, Dataset):
                raise TypeError("Only accept dataset.Dataset Type.")
            dataset.apply(get_vocab)
        vocab.build_vocab()
        return vocab
    
    @staticmethod
    def from_vocab_file(filename):
        vocab_list = read_origin_data(filename, func=str)
        
        vocab = Vocabulary()
        vocab.add_word_list(vocab_list)
        vocab.build_vocab(ignore_sp_token=True)
        return vocab
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            vocab = pickle.load(f)
        return vocab
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def __repr__(self):
        return str(self._word2idx)


def testfrom_vocab_file():
    vocab_list = ['data', 'da', 'care', 'do']
    
    vocab = Vocabulary()
    vocab.add_word_list(vocab_list)
    vocab.build_vocab()
    for word in vocab_list:
        print(vocab.to_index(word))
    vocab = Vocabulary.from_vocab_file("../../bert-base-uncased-vocab.txt")


if __name__ == '__main__':
    data = read_origin_data('../../data/train-v1.1.jsonl', limit=10)
    data = [Instance(**item) for item in data]
    dataset = Dataset(data)
    
    data = read_origin_data('../../data/train-v1.1.jsonl', limit=20)
    data = [Instance(**item) for item in data]
    dataset2 = Dataset(data)
    
    vocab = Vocabulary.from_dataset(dataset, field_names="context")
    
    save_path = "../../vocab_test.pkl"
    vocab.save(save_path)
    
    loaded_vocab = Vocabulary.load(save_path)
    
    vocab.from_dataset(dataset, field_names='context', remain_num=100)
