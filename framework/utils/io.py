#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            io.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/7 16:19    
@Version         1.0 
@Desciption 

'''
import json
import os
import pickle
from typing import List

import yaml

from framework.utils.stat import time_clock


class FindFile:
    
    def __init__(self):
        self.result = []
        self.filter_func = lambda x: True
    
    def walk(self, basedir: str, filter_func=None):
        if filter_func is not None:
            self.filter_func = filter_func
        self.result = []
        self._walk(os.path.abspath(basedir))
    
    def _walk(self, basedir: str):
        """
        the deep search
        :param basedir:
        :return:
        """
        files = os.listdir(basedir)
        
        for file in files:
            file_pth = os.path.join(basedir, file)
            if os.path.isfile(file_pth) and self.filter_func(file_pth):
                self.result.append(file_pth)
            
            else:
                self._walk(basedir)


def find_files(file_name, tgt_dir='.', ext=None):
    result = []
    tgt_dir = os.path.abspath(tgt_dir)
    
    def check_ext(file):
        if ext is None:
            return True
        file_ext = os.path.splitext(file)[-1]
        return file_ext.lower() == ext.lower()
    
    def deep_search(direction):
        file_list = os.listdir(direction)
        for file in file_list:
            if os.path.isfile(os.path.join(direction, file)):
                if check_ext(file) and file.find(file_name) != -1:
                    result.append(os.path.join(direction, file))
            
            else:
                deep_search(os.path.join(direction, file))
    
    deep_search(tgt_dir)
    
    return result


def get_programname(file):
    prgram_filename = os.path.split(file)[1]
    return os.path.splitext(prgram_filename)[0]


def set_base(file, keyword='result'):
    """

    :param file:
    :param keyword:
    :return:
    """
    base_flle = get_programname(file)
    base_dir = os.path.join("../basic/utils", keyword, base_flle)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    return base_dir


def read_yaml(yml_file: str, encoding='utf8'):
    with open(yml_file, encoding=encoding) as f:
        res = yaml.load(f)
        return res


def read_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def read_json(filename):
    def read_once():
        with open(filename, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data
    
    def read_lines():
        data = []
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                line_data = json.loads(line.strip())
                data.append(line_data)
        return data
    
    methods = [read_once, read_lines]
    
    for method in methods:
        try:
            # read a json file once
            result = method()
        except json.decoder.JSONDecodeError as e:
            print("read json method %s failed" % method.__name__)
        except:
            print("read json method %s failed" % method.__name__)
        else:
            return result


def write_json(tgt_file, obj, cls=None):
    with open(tgt_file, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4, cls=cls)


def write_origin_data(tgtfile, data: List, write_type='split'):
    with open(tgtfile, 'w', encoding='utf8') as fout:
        if write_type == 'split':
            for item in data:
                item_str = json.dumps(item)
                fout.write(item_str + '\n')
        else:
            json.dump(data, fout)


@time_clock
def read_origin_data(filename, *, func=None, limit=None):
    """
    read the origin data from multi-lines json
    return the list of json with length of limit

    :param filename: the json filename
    :param limit: only read the limit number of lines of data(if it's None type return )
    :return:
    """
    data_list = []
    
    if func is None:
        func = json.loads
    
    with open(filename, 'r', encoding='utf-8') as f:
        for id, line in enumerate(f.readlines()):
            line = line.strip()
            dic = func(line)
            data_list.append(dic)
            if limit is not None and id > limit - 2:
                break
    return data_list


def write_pkl(filename, obj, user_torch=False):
    if user_torch:
        import torch
        torch.save(obj, filename)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
