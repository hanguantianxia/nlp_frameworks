#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            lstm.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/8/20 20:43    
@Version         1.0 
@Desciption 

'''
from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    
    def __init__(self, *args, batch_first=True, max_seq_len=512, **kwargs):
        r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
        sequence.


        For each element in the input sequence, each layer computes the following
        function:

        .. math::
            \begin{array}{ll} \\
                i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
                f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
                g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
                o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
                c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
                h_t = o_t \odot \tanh(c_t) \\
            \end{array}

        where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
        state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
        is the hidden state of the layer at time `t-1` or the initial hidden
        state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
        :math:`o_t` are the input, forget, cell, and output gates, respectively.
        :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

        In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
        (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
        dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
        variable which is :math:`0` with probability :attr:`dropout`.

        Args:
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
                would mean stacking two LSTMs together to form a `stacked LSTM`,
                with the second LSTM taking in outputs of the first LSTM and
                computing the final results. Default: 1
            bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
                Default: ``True``
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False``
            dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
                LSTM layer except the last layer, with dropout probability equal to
                :attr:`dropout`. Default: 0
            bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``

        Inputs: input, (h_0, c_0)
            - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
              of the input sequence.
              The input can also be a packed variable length sequence.
              See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
              :func:`torch.nn.utils.rnn.pack_sequence` for details.
            - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
              containing the initial hidden state for each element in the batch.
              If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
            - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
              containing the initial cell state for each element in the batch.

              If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


        Outputs: output, (h_n, c_n)
            - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
              containing the output features `(h_t)` from the last layer of the LSTM,
              for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
              given as the input, the output will also be a packed sequence.

              For the unpacked case, the directions can be separated
              using ``output.view(seq_len, batch, num_directions, hidden_size)``,
              with forward and backward being direction `0` and `1` respectively.
              Similarly, the directions can be separated in the packed case.
            - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
              containing the hidden state for `t = seq_len`.

              Like *output*, the layers can be separated using
              ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
            - **c_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
              containing the cell state for `t = seq_len`.

        Attributes:
            weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
                `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
                Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`
            weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
                `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`
            bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
                `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
            bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
                `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`

        .. note::
            All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
            where :math:`k = \frac{1}{\text{hidden\_size}}`

        .. include:: cudnn_persistent_rnn.rst

        Examples::

            >>> rnn = nn.LSTM(10, 20, 2)
            >>> input = torch.randn(5, 3, 10)
            >>> h0 = torch.randn(2, 3, 20)
            >>> c0 = torch.randn(2, 3, 20)
            >>> output, (hn, cn) = rnn(input, (h0, c0))
        """
        super(LSTM, self).__init__()
        self.max_seq_len = max_seq_len
        self.batch_first = batch_first
        self.lstm_encoder = nn.LSTM(*args, batch_first=batch_first, **kwargs)
        num_layers = kwargs.get("num_layers", 1)
        bidrectional = kwargs.get("bidirectional")
        hidden_size = kwargs['hidden_size']
        in_feat = num_layers * (bidrectional + 1) * hidden_size
        dropout = kwargs.get("dropout_out", 0)
        self.h_proj = nn.Linear(in_feat, hidden_size)
        self.c_proj = nn.Linear(in_feat, hidden_size)
        self.output_proj = nn.Linear(hidden_size * (bidrectional + 1), hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor,
                sequence_length: List[int],
                init_hidden_state: Union[None, Tuple[torch.Tensor, torch.Tensor]] = None,
                ):
        """
        lstm encode one step
        :param input:
        :param init_hidden_state:
        :returns:
        :   output: Tensor[batch_size, max_seq_len, seq_len]
        :   (h_0, c_0)
        """
        
        batch_size = input.size(0)
        seq_max_len = int(torch.max(sequence_length).flatten())
        input = input.float()
        if isinstance(sequence_length, torch.Tensor):
            sequence_length = sequence_length.tolist()
        pack_input = torch.nn.utils.rnn.pack_padded_sequence(input,
                                                             sequence_length,
                                                             batch_first=self.batch_first,
                                                             enforce_sorted=False)
        if init_hidden_state is not None:
            packed_encode_output, (h, c) = self.lstm_encoder(pack_input, init_hidden_state)
        
        else:
            packed_encode_output, (h, c) = self.lstm_encoder(pack_input)
        
        encode_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_encode_output,
                                                                  batch_first=self.batch_first,
                                                                  total_length=seq_max_len)
        encode_output = self.output_proj(encode_output)
        encode_output = F.relu(encode_output, inplace=True)
        encode_output = self.dropout(encode_output)
        h = h.permute([1, 0, 2]).reshape(batch_size, -1)
        c = c.permute([1, 0, 2]).reshape(batch_size, -1)
        
        h = self.h_proj(h)
        c = self.c_proj(c)
        
        return encode_output, (h, c)
