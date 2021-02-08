#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            graph.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/9/28 10:18    
@Version         1.0 
@Desciption 

'''

import uuid
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


class GraphManager:
    def __init__(self):
        self.G = nx.Graph()
    
    def adjmat2graph(self, adj_mat, node_len=None):
        """
        convert the adjacency mat into a mat
        :param adj_mat:
        :param node_len:
        :return:
        """
        self.G = nx.Graph()
    
    def update_weight_by_mat(self, adj_mat):
        # todo: Graph Show
        pass
    
    def save_graph(self, filename=None):
        if filename is None:
            filename = str(uuid.uuid1())
        plt.savefig(filename)
        return filename
    
    def add_node(self, node_id):
        self.G.add_node(node_id)
    
    def add_nodes(self, node_ids: List):
        self.G.add_nodes_from(node_ids)
    
    def add_edge(self, start, end):
        self.G.add_edge(start, end)
    
    def add_edges(self, start_end_list: List[Tuple[int, int]]):
        self.G.add_edges_from(start_end_list)
    
    def clear(self):
        self.G.clear()
