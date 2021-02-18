#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            config.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2021/2/9 15:47    
@Version         1.0 
@Desciption 

'''
from abc import abstractmethod

from framework.utils.io import write_json, read_json


class ConfigBase():
    def __init__(self, **kwargs):
        pass
    
    def __repr__(self):
        return str(self.__dict__)
    
    def save(self, path):
        write_json(path, self.__dict__)
    
    @classmethod
    def from_json(cls, json_file):
        config_json = read_json(json_file)
        return cls(**config_json)
    
    def __getitem__(self, item):
        return self.__dict__[item]
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
        setattr(self, key, value)
    
    def get_dict(self):
        return self.__dict__
    
    @classmethod
    def from_args(cls):
        raise NotImplementedError
    
    @classmethod
    def from_dict(cls, dict_ins):
        return cls(**dict_ins)
    
    @abstractmethod
    def self_check(self):
        """
        check the config is reasonable, written by users.
        
        """
