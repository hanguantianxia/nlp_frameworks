#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            static_embedding.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/7/10 17:00    
@Version         1.0 
@Desciption 

'''
from framework.embedding.static_embedding import *
import unittest

from test.framework.test_util import generate_sample_dataset, generate_tokned_dataset


class TestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = generate_sample_dataset()
        self.token_dataset = generate_tokned_dataset(100)
        self.vocab_path = r'G:\Model\word2vec\eng\glove.6B.300d.txt'
        self.test_ids = [1, 2, 3, 4, 5, 7]
        self.vocab = Vocabulary.from_dataset(self.token_dataset, field_names="token_context")

    def test_parse_func(self):

        embed = default_parse_func(self.vocab_path, self.vocab, 300, limit=30)
        vocab_size = len(self.vocab)
        hidden_size = 300

        self.assertIsInstance(embed, torch.Tensor)
        self.assertEqual(embed.size(), (vocab_size, hidden_size))

    def test_StaticEmbed_from_pretrain(self):
        limit = 30
        eps = 1E-5
        embed = StaticEmbedding.from_pretrain(self.vocab_path,
                                              self.vocab,
                                              300,
                                              padding_idx=0,
                                              freeze=True,
                                              limit=limit)
        word_dict = read_word2vec(self.vocab_path, limit=limit)

        for word, vec in word_dict.items():
            token_id = self.vocab.to_index(word)
            if token_id == self.vocab.unk_token_id:
                continue
            token_id_pt = torch.LongTensor([token_id])
            embed_vec = embed(token_id_pt)
            embed_vec_arr = np.array(embed_vec).flatten()
            vec_arr = np.array(vec).flatten()

            error = np.sum((vec_arr - embed_vec_arr) ** 2) / vec_arr.size
            self.assertLess(error, eps)

    def test_StaticEmbed_normal(self):
        embed = StaticEmbedding(len(self.vocab), 300)

        for token_id in range(len(self.vocab)):
            input_ids = torch.LongTensor([token_id])
            vect = embed(input_ids)
            self.assertEqual(vect.size(), (1, 300))
