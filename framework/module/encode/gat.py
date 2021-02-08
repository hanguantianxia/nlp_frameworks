#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            gat.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/9/5 15:41    
@Version         1.0 
@Desciption 

'''
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReverseGraphAttentionLayer(nn.Module):
    """
    Given the attention matrix, output the data
    """
    
    def __init__(self, input_size: int = 768,
                 infer_hidden_size=768,
                 slope=0.2,
                 dropout=0.1,
                 attn_dropout=0,
                 concat=False):
        super(ReverseGraphAttentionLayer, self).__init__()
        self.hidden_size = input_size
        self.infer_hidden_size = infer_hidden_size
        self.slope = slope
        self.dropout = dropout
        self.concat = concat
        self.attn_dropout = attn_dropout
        
        self.W = nn.Parameter(torch.zeros(size=(self.hidden_size, self.infer_hidden_size)))
        nn.init.kaiming_normal_(self.W.data)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * self.infer_hidden_size, 1)))
        nn.init.kaiming_normal_(self.a.data)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.attn_dropout_layer = nn.Dropout(p=self.attn_dropout)
        # print(self.a.shape)  torch.Size([16, 1])
        self.leakyrelu = nn.LeakyReLU(self.slope)
        
        self.inf = 9e15
    
    def forward(self, input: torch.Tensor,
                node_length: torch.Tensor,
                graph_structure: torch.Tensor):
        """

        :param input: the input matrix of h_i [batch_size, max_node_len, hidden_size]
        :param node_length: the node length of every batch, [batch_size, max_node_len]
        :param graph_structure: Adjacency matrix of the graph if we have adj [batch_size,max_node_len,max_node_len]
        :return:
        : graph node vector: torch.Tensor [batch_size, max_node_Len, hidden_size]
        : graph structure matrix: torch.Tensor [batch_size, max_node_len, hidden_size]
        todo:
            1. adj
        """
        node_vec = torch.matmul(graph_structure, input)
        if self.concat:
            node_vec = F.elu(node_vec)
        
        h = torch.matmul(node_vec, self.W)  # h [batch_size, max_node_Len,infer_hidden_size]
        max_node_len = h.size(1)
        
        a_input = torch.cat([h.repeat_interleave(max_node_len, dim=1),
                             h.repeat(1, max_node_len, 1)], dim=-1)
        # a_input_size [batch_size, max_node_len*max_node_len, infer_hidden_size*2]
        a_input = a_input.reshape(-1, max_node_len, max_node_len, self.infer_hidden_size * 2)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        zero_vec = -self.inf * torch.ones_like(e, device=self.device)
        assert node_length is not None, \
            "Must input node length of every batch when you don't input the adjacency matrix"
        bias = torch.bmm(node_length.unsqueeze(dim=2), node_length.unsqueeze(dim=1))
        # keep Adjacency relation like the structure
        attn_scores = torch.where(graph_structure > 1E-6, e, zero_vec)
        attn_scores = torch.where(bias > 1E-6, attn_scores, zero_vec)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attention_drop = self.attn_dropout_layer(attn_scores)
        
        return node_vec, attention_drop
    
    @property
    def device(self):
        return self.W.data.device


class ReverseGATLayer(nn.Module):
    
    def __init__(self, input_size=768,
                 hidden_size=768,
                 dropout=0.2,
                 attn_dropout=0,
                 nheads=4,
                 slope=0.2):
        super(ReverseGATLayer, self).__init__()
        self.dropout = dropout
        self.nhead = nheads
        self.slope = slope
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.attentions = [
            ReverseGraphAttentionLayer(self.input_size, self.hidden_size, dropout=dropout, attn_dropout=attn_dropout,
                                       slope=slope, concat=True)
            for _ in range(nheads)]
        self.output_proj = nn.Linear(self.input_size * nheads, self.hidden_size)
        # input the hidden layer
        for i, attention in enumerate(self.attentions):
            self.add_module("Reverse_Graph_Attention_Head_{}".format(i), attention)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, input: torch.Tensor,
                node_length: torch.Tensor,
                graph_structure: List[torch.Tensor]):
        """

        :param input: torch.Tensor of node vector [batch_size, max_node_len, hidden_size]
        :param node_length: the length of nodes in a batch, [batch_size, 1]
        :param graph_structure: a list of torch tensor of graph structure
        :return:
        """
        assert len(graph_structure) == self.nhead, "the graph structure must have the same length as of nheads "
        input = self.dropout(input)
        batch_size, max_node_len, _ = input.size()
        
        if node_length.size() != (batch_size, max_node_len):
            node_length = self.make_node_len_mat(node_length, max_node_len)
        
        attn_res_list = [attn(input, node_length, graph_structure[attn_id]) for attn_id, attn in
                         enumerate(self.attentions)]
        node_vec_list = [item[0] for item in attn_res_list]
        attn_vec_list = [item[1] for item in attn_res_list]
        
        node_vector = torch.cat(node_vec_list, dim=-1)
        node_vector = self.output_proj(node_vector)
        
        return node_vector, attn_vec_list
    
    def make_node_len_mat(self, node_len, max_node_len=None):
        """

        :param node_len: [batch_size, 1]
        :return:  [batch_size, 1]
        """
        batch_size = node_len.size(0)
        if max_node_len is None:
            max_node_len = max(node_len)
        
        node_len_mat = torch.zeros((batch_size, max_node_len), device=self.device)
        for batch_id, length in enumerate(node_len):
            node_len_mat[batch_id, :length] = 1
        
        return node_len_mat
    
    @property
    def device(self):
        return self.attentions[0].device


class BiGraphAttentionLayer(nn.Module):
    
    def __init__(self, hidden_size: int = 768,
                 infer_hidden_size=768,
                 slope=0.2,
                 dropout=0.1,
                 attn_dropout=0,
                 concat=False):
        super(BiGraphAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.infer_hidden_size = infer_hidden_size
        self.slope = slope
        self.dropout = dropout
        self.concat = concat
        self.attn_dropout = attn_dropout
        
        self.W = nn.Parameter(torch.zeros(size=(self.hidden_size, self.infer_hidden_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * self.infer_hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.attn_dropout_layer = nn.Dropout(p=self.attn_dropout)
        # print(self.a.shape)  torch.Size([16, 1])
        self.leakyrelu = nn.LeakyReLU(self.slope)
        
        self.inf = 9e15
    
    def node2graph(self, input: torch.Tensor, node_length: torch.Tensor = None, adj=None):
        """

        :param input:
        :param node_length:
        :param adj:
        :return:
        """
        
        assert adj is not None or node_length is not None, \
            "Must input node length of every batch when you don't input the adjacency matrix"
        
        h = torch.matmul(input, self.W)
        
        max_node_len = h.size()[1]
        
        a_input = torch.cat([h.repeat_interleave(max_node_len, dim=1),
                             h.repeat(1, max_node_len, 1)], dim=-1)
        # a_input_size [batch_size, max_node_len*max_node_len, infer_hidden_size*2]
        a_input = a_input.reshape(-1, max_node_len, max_node_len, self.infer_hidden_size * 2)
        attn_scores = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        zero_vec = -self.inf * torch.ones_like(attn_scores, device=self.device)
        if adj is not None:
            attn_scores = torch.where(adj > 0, attn_scores, zero_vec)
        if node_length is not None:
            bias = torch.bmm(node_length.unsqueeze(dim=2), node_length.unsqueeze(dim=1))
            if torch.sum(node_length, dtype=torch.int64) != 1:
                bias = self.attn_dropout_layer(bias)
            attn_scores = torch.where(bias > 0, attn_scores, zero_vec)
        # attention [batch_size, max_len_node, max_len_node]
        attn_scores = F.softmax(attn_scores, dim=-1)
        nodes_vector = torch.matmul(attn_scores, h)
        if self.concat:
            nodes_vector = F.elu(nodes_vector)
        
        return nodes_vector, attn_scores
    
    def graph2node(self, input: torch.Tensor,
                   node_length: torch.Tensor,
                   graph_structure: torch.Tensor):
        """

        :param input:
        :param node_length:
        :param graph_structure:
        :return:
        """
        
        node_vec = torch.matmul(graph_structure, input)
        if self.concat:
            node_vec = F.elu(node_vec)
        
        h = torch.matmul(node_vec, self.W)  # h [batch_size, max_node_Len,infer_hidden_size]
        max_node_len = h.size(1)
        
        a_input = torch.cat([h.repeat_interleave(max_node_len, dim=1),
                             h.repeat(1, max_node_len, 1)], dim=-1)
        # a_input_size [batch_size, max_node_len*max_node_len, infer_hidden_size*2]
        a_input = a_input.reshape(-1, max_node_len, max_node_len, self.infer_hidden_size * 2)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        zero_vec = -self.inf * torch.ones_like(e, device=self.device)
        assert node_length is not None, \
            "Must input node length of every batch when you don't input the adjacency matrix"
        bias = torch.bmm(node_length.unsqueeze(dim=2), node_length.unsqueeze(dim=1))
        # keep Adjacency relation like the structure
        attn_scores = torch.where(graph_structure > 1E-6, e, zero_vec)
        attn_scores = torch.where(bias > 1E-6, attn_scores, zero_vec)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.attn_dropout_layer(attn_scores)
        
        return node_vec, attn_scores
    
    def forward(self, input: torch.Tensor,
                node_length: torch.Tensor = None,
                graph_struct=None,
                direction='f'):
        """

        :param input:
        :param node_length:
        :param graph_struct:
        :param direction:
        :return:
        """
        if direction.find('f') != -1:
            nodes_vector, attn_scores = self.node2graph(input, node_length, graph_struct)
        elif direction.find('b') != -1:
            nodes_vector, attn_scores = self.graph2node(input, node_length, graph_struct)
        else:
            raise TypeError("direction only accept the forward('f') and backward('b'), but you input %s" % direction)
        
        return nodes_vector, attn_scores
    
    @property
    def device(self):
        return self.W.device


class BiGATLayer(nn.Module):
    def __init__(self, input_size=768,
                 hidden_size=768,
                 dropout=0.2,
                 attn_dropout=0,
                 nheads=6,
                 slope=0.2):
        super(BiGATLayer, self).__init__()
        self.dropout = dropout
        self.nhead = nheads
        self.slope = slope
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.attentions = [
            BiGraphAttentionLayer(self.input_size, self.hidden_size, dropout=dropout, attn_dropout=attn_dropout,
                                  slope=slope, concat=True)
            for _ in range(nheads)]
        self.output_proj = nn.Linear(self.input_size * nheads, self.hidden_size)
        # input the hidden layer
        for i, attention in enumerate(self.attentions):
            self.add_module("Reverse_Graph_Attention_Head_{}".format(i), attention)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, input: torch.Tensor,
                node_length: torch.Tensor,
                graph_structure: List[torch.Tensor], direction):
        """

        :param input:
        :param node_length:
        :param graph_structure:
        :return:
        """
