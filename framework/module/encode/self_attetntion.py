#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            self_attetntion.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/11/5 20:38    
@Version         1.0 
@Desciption 

'''
import copy
import math
from typing import Union, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList


class MultiheadAttetion(nn.Module):
    
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 bias=False,
                 add_bias_kv=False,
                 kdim=None,
                 vdim=None):
        super(MultiheadAttetion, self).__init__()
        
        self.embed_dim = embed_dim
        self.kdim = kdim if vdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.embed_dim == self.kdim and self.vdim == embed_dim
        
        self.num_heads = num_heads
        self.head_dims = embed_dim // num_heads
        assert self.head_dims * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj_weight = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj_weight = nn.Linear(embed_dim, self.kdim, bias=add_bias_kv)
        self.v_proj_weight = nn.Linear(embed_dim, self.vdim, bias=add_bias_kv)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                query,
                key,
                value,
                attn_mask=None,
                attn_method=None):
        """

        :param query: torch.Tensor[batch_size, max_seq_len, hidden_size]
        :param key: torch.Tensor[batch_size, max_seq_len, hidden_size]
        :param value: torch.Tensor[batch_size, max_seq_len, hidden_size]
        :param attn_mask: torch.Tensor[batch_size*nheads ,max_seq_len, max_seq_len]
        :return:
        """
        if self._qkv_same_embed_dim is True:
            
            q_proj = self.q_proj_weight(query)  # [batch_size, max_seq_len, hidden_size]
            k_proj = self.q_proj_weight(key)  # [batch_size, max_seq_len, hidden_size]
            v_proj = self.q_proj_weight(value)  # [batch_size, max_seq_len, hidden_size]
            
            q_proj = torch.stack(q_proj.split(self.head_dims, dim=-1),
                                 dim=1)  # [batch_size, num_heads, max_seq_len, hidden_size/num_heads]
            k_proj = torch.stack(k_proj.split(self.head_dims, dim=-1),
                                 dim=1)  # [batch_size, num_heads, max_seq_len, hidden_size/num_heads]
            v_proj = torch.stack(v_proj.split(self.head_dims, dim=-1),
                                 dim=1)  # [batch_size, num_heads, max_seq_len, hidden_size/num_heads]
            
            attn_logit = torch.matmul(q_proj, k_proj.permute([0, 1, 3, 2])) / math.sqrt(
                self.head_dims)  # [batch_size, num_heads, max_seq_len, max_seq_len]
            if attn_mask is not None:
                attn_logit = self.attention_mask(attn_logit, attn_mask, attn_method)
            attn_logit = self.dropout(attn_logit)
            attn_scores = attn_logit.softmax(dim=-1)
            
            attn_res = torch.matmul(attn_scores, v_proj)  # [batch_size, num_heads, max_seq_len, hidden_size/num_heads]
            attn_res = attn_res.permute([0, 2, 1, 3]).reshape_as(query)  # [batch_size, max_seq_len, hidden_size]
            output = self.output_proj(attn_res)
            
            return output, attn_scores
        
        
        else:
            raise TypeError("k,q,v must be the same size")
    
    def attention_mask(self, attn_logit,
                       attn_mask: torch.Tensor,
                       mask_method=None):
        """

        1. add: abandon = -inf, keep = 0,
        2. multi: abandon = 0, keep = 1, 0<other<1 or 1<other<inf
        3. mask: abandon = 0, keep = 1 or bool true is keep

        :param attn_logit:
        :param attn_mask:
        :param mask_method: offer different method to use mask, now available methods are ['mask', 'add', 'multiply']
        :return: mask torch.Tensor[batch_size, num_heads,max_seq_len, max_seq_len]
        """
        
        def mask(logit: torch.Tensor, attn_mask: torch.Tensor):
            attn_mask = attn_mask > 1e-6
            attn_mask = (attn_mask.float() - 1) * 1000000
            logit = logit + attn_mask
            return logit
        
        def add(logit: torch.Tensor, attn_mask: torch.Tensor):
            logit = logit + attn_mask
            return logit
        
        def multi(logit: torch.Tensor, attn_mask: torch.Tensor):
            logit = logit * attn_mask
            logit = mask(logit, attn_mask)
            return logit
        
        mask_methods = {
            mask.__name__: mask,
            add.__name__: add,
            multi.__name__: multi
        }
        
        if attn_mask is None:
            return None
        
        batch_size, num_heads, max_seq_len, _ = attn_logit.size()
        
        if attn_mask.size() == (batch_size * self.num_heads, max_seq_len, max_seq_len):
            attn_mask = attn_mask.reshape(batch_size, self.num_heads, max_seq_len, max_seq_len)
        elif attn_mask.size() == (batch_size, self.num_heads, max_seq_len, max_seq_len):
            pass
        elif attn_mask.size() == (batch_size, max_seq_len, max_seq_len):
            attn_mask = attn_mask.unsqueeze(dim=1).repeat([1, self.num_heads, 1, 1])
        else:
            raise TypeError(
                "The mask shape only accept [b, h, s,s], [b,s,s], [b*h,s,s] b=batch_size, h=num of heads, s=max_seq_len")
        # modify the
        
        # if mask_method not in mask_methods:
        #     print("Warning: The {} not in {}. Now we use {} method".format(str(mask_method),list(mask_methods.keys()), add.__name__))
        mask_method_func = mask_methods.get(mask_method, mask)
        return mask_method_func(attn_logit, attn_mask)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device


class SelfAttentionLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 hidden_size=None,
                 attn_prob_dropout=0.0,
                 hidden_dropout=0.1,
                 bias=False,
                 add_bias_kv=False,
                 kdim=None,
                 vdim=None):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = embed_dim * 4 if hidden_size is None else hidden_size
        self.multihead = MultiheadAttetion(embed_dim, num_heads, attn_prob_dropout, bias, add_bias_kv, kdim, vdim)
        self.inner_proj = nn.Linear(embed_dim, self.hidden_size)
        self.outer_proj = nn.Linear(self.hidden_size, embed_dim)
        
        self.attn_prob_dropout = attn_prob_dropout
        self.hidden_dropout_rate = hidden_dropout
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self,
                query,
                key=None,
                value=None,
                attn_mask=None,
                attn_method=None,
                return_attn=True):
        """

        :param query:
        :param key:
        :param value:
        :param return_attn:
        :return:
        """
        if key is None and value is None:
            key = value = query
        attn_output, attn_scores = self.multihead(query, key, value, attn_mask, attn_method)
        
        attn_output = self.norm(self.hidden_dropout(attn_output) + query)
        
        proj_output = self.outer_proj(F.relu(self.hidden_dropout(self.inner_proj(attn_output))))
        
        self_out = self.norm(attn_output + self.hidden_dropout(proj_output))
        if return_attn:
            return self_out, attn_scores
        else:
            return self_out


class SelfAttention(nn.Module):
    
    def __init__(self, encoder_layer: nn.Module, num_layers, norm=None):
        super(SelfAttention, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm
    
    def forward(self, src: torch.Tensor, attn_mask=None, attn_method=None, return_attn=True):
        output = src
        attn_scores_list = []
        
        for mod in self.layers:
            output, attn_scores = mod(output, attn_mask=attn_mask, attn_method=attn_method, return_attn=return_attn)
            attn_scores_list.append(attn_scores)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output, attn_scores_list


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def self_attention_output(doc_structure_output: Tuple) -> Tuple[torch.Tensor, List[torch.Tensor], Union[None, Dict]]:
    """

    :param doc_structure_output: a tuple of torch.Tensor [batch_size, num_heads, max_seq_len, hidden_size]
    :return:
    """
    output, attn_scores = doc_structure_output
    attn_score_list = attn_scores.split(1, dim=1)
    attn_score_list = [item.squeeze(dim=1) for item in attn_score_list]
    return output, attn_score_list, {}


def self_attention_input(edus_vector, node_length, adj_mats):
    adj_mats = torch.stack(adj_mats, dim=1)
    return edus_vector, None, None, adj_mats


if __name__ == '__main__':
    input = torch.rand(8, 4, 128)
    mask = torch.ones((8, 8, 4, 4)).tril()
    
    model = SelfAttentionLayer(128, 8, bias=False)
    output, df = model(input, attn_mask=mask)
