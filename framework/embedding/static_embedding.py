#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            static_embedding.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/30 20:36    
@Version         1.0 
@Desciption 

'''
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from framework.basic.vocablary import Vocabulary


class StaticEmbedding(nn.Module):
    
    def __init__(self, vocab_size: int = None,
                 hidden_dim: int = None,
                 embed: torch.Tensor = None,
                 padding_idx=None,
                 **kwargs):
        super(StaticEmbedding, self).__init__()
        if vocab_size is None and hidden_dim is None and embed is None:
            raise TypeError("Must input vocab size and hidden dim or embed")
        
        if embed is None:
            assert all((vocab_size, hidden_dim)), "Must input vocab size and hidden_dim at the same time"
            self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(embed, **kwargs)
    
    def forward(self, input) -> torch.Tensor:
        return self.embedding(input)
    
    @staticmethod
    def from_pretrain(files,
                      vocab: Vocabulary,
                      hidden_size: int,
                      parse_func=None,
                      limit=None,
                      **kwargs):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        :param files: vector files
        :param vocab: Vocabulary class
        :param hidden_size: hidden dimension
        :param parse_func: parse vocabulary function, its interface should be
        parse_func(file, vocab, hidden_size: int, limit=None) -> torch.Tensor:
        file means vector file, hidden_size
        :param limit: the limitation of reading the file
        :param kwargs:
            include:
            freeze (boolean, optional) – If True, the tensor does not get updated in the learning process. Equivalent to embedding.weight.requires_grad = False. Default: True
            padding_idx (int, optional) – See module initialization documentation.
            max_norm (float, optional) – See module initialization documentation.
            norm_type (float, optional) – See module initialization documentation. Default 2.
            scale_grad_by_freq (boolean, optional) – See module initialization documentation. Default False.
            sparse (bool, optional) – See module initialization documentation.
        :return:
        """
        if parse_func is None:
            embed_tensor = default_parse_func(files, vocab, hidden_size, limit=limit)
        else:
            embed_tensor = parse_func(files, vocab)
        return StaticEmbedding(embed=embed_tensor, **kwargs)


def default_parse_func(file, vocab, hidden_size: int, limit=None) -> torch.Tensor:
    vocab_size = len(vocab)
    embed_arr = np.random.randn(vocab_size, hidden_size)
    
    with open(file, 'r', encoding='utf8') as f:
        for line_id, line in enumerate(f):
            if limit is not None and line_id > limit:
                break
            if line_id == 0:
                first_line = line
                try:
                    _, hidden_len = first_line.split(' ')
                except:
                    pass
            
            line_list = line.strip().split(' ')
            word = line_list[0]
            vect = line_list[1:]
            if len(vect) == hidden_size:
                vect = [float(item) for item in vect]
                word_id = vocab.to_index(word)
                if word_id != vocab.unk_token_id:
                    embed_arr[word_id, :] = vect
    
    embed_pt = torch.tensor(embed_arr)
    return embed_pt


def read_word2vec(file, limit=None) -> Dict[str, list]:
    word_dict = {}
    with open(file, 'r', encoding='utf8') as f:
        for line_id, line in enumerate(f):
            if line_id > limit:
                break
            if line_id == 0:
                first_line = line
                try:
                    _, hidden_len = first_line.split(' ')
                except:
                    pass
            
            line_list = line.strip().split(' ')
            word = line_list[0]
            vect = line_list[1:]
            vect = [eval(item) for item in vect]
            word_dict[word] = vect
    
    return word_dict


if __name__ == '__main__':
    filename = r'E:\Model\word2vec\eng\glove.6B.300d.txt'
    a = read_word2vec(filename, 10)
