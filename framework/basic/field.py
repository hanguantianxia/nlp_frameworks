#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            field.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/26 14:07    
@Version         1.0 
@Desciption 

'''
import copy
from abc import abstractmethod
from collections import Iterable
from typing import Union, List


class Field:
    special_types = ["str_list", "str_dict"]
    """
    to manager the Filed of Dataset
    """
    
    def __init__(self, name,
                 init_list,
                 special_type=None,
                 padder=None
                 ):
        # if isinstance(init_list, list) and len(init_list)!=0 and  (not isinstance(init_list[0], list)):
        #     init_list = [init_list]
        if not isinstance(init_list, list):
            init_list = [init_list]
        if len(init_list) == 0:
            raise RuntimeError("Empty filed content is not allowed.")
        
        type_set = set([type(item) for item in init_list])
        if len(type_set) != 1:
            print("Warning:the init list should have the same type.Now {} have {}".format(name, type_set))
        
        self._data = init_list
        if padder is None:
            padder = DefaultPadder
        assert issubclass(padder, Padder), "The padder should be the subclass of Padder."
        
        self.name = name
        self._dtype = type(init_list[0])
        self._string_max_len = None
        self._list_max_len = None
        
        # for the batch generate process
        self.padder = padder
    
    def __getitem__(self, item: Union[List[int], int, slice]):
        if isinstance(item, slice):
            return self._data[item]
        elif isinstance(item, int):
            return self._data[item]
        elif isinstance(item, Iterable):
            data = []
            for data_item in data:
                assert isinstance(data_item, int), "Field Only accept list of int"
                data.append(self._data[data_item])
            return data
        else:
            raise TypeError("Field getitem only accept one of List[int], int, or slice Type")
    
    @property
    def string_max_len(self):
        if self._string_max_len is None:
            self._string_max_len = self._compute_max_str_len()
        return self._string_max_len
    
    @property
    def list_max_len(self):
        if self._list_max_len is None:
            self._list_max_len = self._compute_max_list_len()
        return self._list_max_len
    
    @property
    def field_name(self):
        return self.name
    
    @field_name.setter
    def field_name(self, new_field_name):
        self.name = new_field_name
    
    @property
    def dtype(self):
        return self._dtype
    
    @dtype.setter
    def dtype(self, data_type):
        self._dtype = data_type
    
    @property
    def data(self):
        return self._data
    
    def _compute_max_str_len(self):
        return max([len(str(item)) for item in self])
    
    def append(self, item) -> None:
        # def update_str_max_len():
        #     self._string_max_len = max(len(str(item)), self.string_max_len)
        #
        # update_str_max_len()
        self._data.append(item)
    
    def _compute_max_list_len(self):
        if isinstance(self[0], list):
            return max([len(item) for item in self])
        return 0
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    
    def __repr__(self):
        return str(self._data)
    
    def __setitem__(self, key, value):
        self._data[key] = value


class Padder:
    """
    This class is to pad the sequence to the same length
    """
    
    def __init__(self, pad_token="<pad>"):
        self.pad_token = pad_token
    
    def set_pad_token(self, pad_token):
        self.pad_token = pad_token
    
    def get_pad_token(self):
        return self.pad_token
    
    @abstractmethod
    def __call__(self, content: Field, max_len=None, in_place=True):
        """
        Only support
        :param content: Dataset
        :param field_name: the field names
        :return:
        """
        raise NotImplementedError


class DefaultPadder(Padder):
    
    def __init__(self, pad_token="<pad>"):
        super(DefaultPadder, self).__init__(pad_token)
    
    def __call__(self, content: Field, max_len=None, in_place=True):
        if max_len is not None and not isinstance(max_len, int):
            raise TypeError("The max length max be the int type")
        seq_max_len = content.list_max_len if max_len is None else max_len
        
        if not in_place:
            content = copy.deepcopy(content)
        
        def pad(item):
            if isinstance(item, list):
                assert isinstance(item[0], str), \
                    "This field %s cannot be padded. Only support list of string." % content.name
                extend_list = [self.pad_token] * (seq_max_len - len(item))
                item.extend(extend_list)
            elif isinstance(item, str):
                extend_list = [self.pad_token] * (seq_max_len - len(item))
                extend_token_str = " ".join(extend_list)
                
                item += " " + extend_token_str
            else:
                raise TypeError("Only support list of string of string")
            
            return item
        
        for item_id, item in enumerate(content):
            padded_item = pad(item)
            content[item_id] = padded_item
        
        return content
