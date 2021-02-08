#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            stat.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/7 16:21    
@Version         1.0 
@Desciption 

'''

import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_bar(height, x=None):
    if x is None:
        x = list(range(len(height)))
    
    plt.bar(x, height)
    plt.show()


def show_curve(x, y):
    assert len(x) == len(y)
    
    plt.plot(x, y)
    plt.show()


class TimeManager():
    
    def __init__(self):
        self.clocks = defaultdict(list)
        self.pre_time = None
        self.pre_name = None
    
    def start(self, name='default'):
        self.pre_name = name
        self.pre_time = time.time()
    
    def end(self):
        cur_time = time.time()
        self.clocks[self.pre_name].append(cur_time - self.pre_time)
    
    def statistic(self, method='mean'):
        statistic_res = {}
        for k, v in self.clocks.items():
            if method == 'mean':
                v = np.array(v).mean()
                statistic_res[k] = float(v)
        return statistic_res
    
    def __repr__(self):
        return str(self.statistic())


def time_clock(func):
    def compute_time_clock(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Implement function %s, using %.2f s" % (func.__name__, end - start))
        return res
    
    return compute_time_clock


def show_hist(data, title="", xlabel="", ylabel="", bins=10, range='auto', show=True, y_range=(0, 1)):
    if range == 'auto':
        data = np.array(data)
        range_size = (np.min(data), np.max(data))
    else:
        range_size = range
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    prob, left, rectangle = plt.hist(x=data,
                                     bins=bins,
                                     color="steelblue",
                                     edgecolor="black",
                                     range=range_size,
                                     density=True,
                                     stacked=True
                                     )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(y_range)
    if show:
        plt.show()
    return prob, left, rectangle


def hist_analyse(data, title="", xlabel="", ylabel="", bins=10, show=True, kind="bar"):
    data = pd.Series(data)
    bins_obj = data.value_counts(bins=bins)
    if show:
        bins_obj.plot(kind=kind)
        plt.show()
    
    prob = bins_obj.to_numpy(copy=True)
    prob = prob / np.sum(prob)
    interval = bins_obj.keys().to_tuples()
    
    return prob, interval
