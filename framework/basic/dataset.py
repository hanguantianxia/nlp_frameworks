#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            dataset.py.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/6/17 15:47    
@Version         1.0 
@Desciption 

'''
__all__ = ['Instance', 'Dataset']

import copy
import pickle
from collections import UserDict, Iterable, Callable, defaultdict
from typing import List, Union, Dict

from tqdm import tqdm

from framework.basic.field import Field
from framework.basic.tokenizer import EnglishTokenizer
from framework.basic.utils.inner_utils import pretty_table_printer
from framework.utils.io import read_origin_data


class Instance(UserDict):
    
    def __init__(self, instance=None, **fileds):
        if isinstance(instance, Instance) or isinstance(instance, dict):
            super(Instance, self).__init__(**instance)
        else:
            super(Instance, self).__init__(fileds)
        
        for key, value in fileds.items():
            setattr(self, key, value)
    
    def add_field(self, field_name, field):
        self[field_name] = field
    
    def remove_filed(self, field_name):
        if field_name in self:
            self.pop(field_name)
    
    def set_field(self, field_name, field):
        self.add_field(field_name, field)
    
    def __repr__(self):
        return str(pretty_table_printer(self))


class Dataset():
    
    def __init__(self, data: Union[None, List, Dict] = None, key_uncased=False):
        """

        :param data:
        """
        self.key_uncased = key_uncased
        self._data_container = {}  # table
        if data is not None:
            if isinstance(data, dict):
                length_list = []
                for key, value in data.items():
                    length_list.append(len(value))
                length_set = set(length_list)
                assert len(length_set) == 1, \
                    "Data arrays must have the same length. Now %s" % \
                    {key: length for key, length in zip(data.keys(), length_list)}
                
                for key, value in data.items():
                    if self.key_uncased:
                        key = key.lower()
                    self.add_field(key, value)
            elif isinstance(data, list):
                for ins in data:
                    try:
                        ins = Instance(ins)
                    except:
                        raise TypeError("This type cannot be convert to Instance class.")
                    # assert isinstance(ins, Instance), "Data must be Instance type, not {}.".format(type(ins))
                    self.append(ins)
            else:
                raise TypeError("the data type must be the dict or list[instance]")
    
    def append(self, instance: Union[Instance, None], is_copy=False):
        if isinstance(instance, Instance):
            self._append_instance(instance, is_copy)
        elif isinstance(instance, Dataset):
            self.add_dataset(instance)
        else:
            raise TypeError("'append' method only accept the Instance and Dataset!")
    
    def _append_instance(self, instance: Instance, is_copy=False):
        if len(self._data_container) == 0:
            for key, value in instance.items():
                value = [value]
                if self.key_uncased:
                    key = key.lower()
                self.add_field(key, value)
        else:
            assert len(self._data_container) == len(instance), \
                "instance must have the same length %d as dataset %d" \
                % (len(self._data_container), len(instance))
            if self.key_uncased:
                ori_keys = [key.lower() for key in list(self._data_container.keys())]
                new_keys = [key.lower() for key in list(instance.keys())]
                assert set(ori_keys) == set(new_keys), \
                    "instance must have the same field as dataset"
            else:
                assert set(self._data_container.keys()) == set(instance.keys()), \
                    "instance must have the same field as dataset"
            for key, value in self._data_container.items():
                if self.key_uncased:
                    key = key.lower()
                if is_copy:
                    instance = copy.deepcopy(instance)
                value.append(instance[key])
    
    def add_field(self, field_name, field):
        
        if self.key_uncased:
            field_name = field_name.lower()
        if field_name in self._data_container:
            UserWarning("We have already have field %s, now we will overwrite this write" % field_name)
        
        if not isinstance(field, Field):
            field = Field(field_name, field)
        if not (len(field) == len(self) or len(self) == 0):
            string = "Insert field must have the same length as dataset. Now %d vs %d" \
                     % (len(field), len(self))
        self._data_container[field.name] = field
    
    @property
    def field_list(self):
        return list(self._data_container.values())
    
    @property
    def field_name_list(self):
        return list(self._data_container.keys())
    
    def remove_field(self, field_name):
        return self._data_container.pop(field_name)
    
    def rename_field(self, src_field_name, tgt_field_name):
        """

        :param src_field_name:
        :param tgt_field_name:
        :return:
        """
        assert src_field_name in self, "The source field name %s is not in this dataset." % src_field_name
        field = self.remove_field(src_field_name)
        field.field_name = tgt_field_name
        self.add_field(tgt_field_name, field)
    
    def split(self, split_method: Union[None, callable] = None, partitions: Union[list, None] = None):
        """
        callable method:
        Input the whole datset,select the instance to plist
        Output the select ids
        :param split_method:
        :return:
        """
        if split_method != None:
            assert callable(split_method), "split method must be a callable class of a function!"
            return split_method(self)
        else:
            if partitions is None:
                raise TypeError(
                    "paritions cann't not be None, if don't give the split method. Default method is random sample which need partition.")
            # todo: 1. code the default split method
    
    def filter(self, judgement_func, deepCopy=False):
        
        dataset = Dataset()
        for item in self:
            if judgement_func(item):
                dataset.append(item)
        
        if deepCopy:
            dataset = copy.deepcopy(dataset)
        
        return dataset
    
    def __len__(self):
        if len(self._data_container) == 0:
            return 0
        field = next(iter(self._data_container.values()))
        return len(field)
    
    def __getitem__(self, item):
        """
        1. item type is int: reuturn a instance
        2. item type is slice,return a dataset
        3. item type is list,return a dataset
        :param item:
        :return:
        """
        if isinstance(item, int):
            if item >= len(self) or item < 0:
                raise IndexError("index out of range.")
            target_data = {}
            for key, value in self._data_container.items():
                target_data[key] = value[item]
            return Instance(**target_data)
        
        elif isinstance(item, slice):
            if item.start is not None and (item.start >= len(self) or item.start <= -len(self)):
                raise RuntimeError(f"Start index {item.start} out of range 0-{len(self) - 1}")
            target_data = {}
            for key, value in self._data_container.items():
                target_data[key] = value[item]
            return Dataset(target_data)
        
        elif isinstance(item, str):
            if self.key_uncased:
                item = item.lower()
            return self._data_container[item]
        
        elif isinstance(item, Iterable):
            dataset = Dataset()
            for i in item:
                dataset.append(self[i])
            return dataset
        
        else:
            raise TypeError("Unexpect type {} for idx in __getitem__ method".format(type(item)))
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            if self.key_uncased:
                key = key.lower()
            assert isinstance(value, Field), "value must be the instance of Field."
            assert len(value) == len(self) or len(self) == 0, ""
            if key in self:
                self.remove_field(key)
            self.add_field(key, value)
        
        else:
            raise TypeError("Unexpect type {} for idx in __getitem__ method".format(type(key)))
    
    def __contains__(self, item):
        return item in self._data_container
    
    def __repr__(self):
        return str(pretty_table_printer(self))
    
    def __iter__(self):
        for sample_id in range(len(self)):
            yield self[sample_id]
    
    def apply(self, func, new_field_names: Union[List[str], None] = None, show_process=False):
        """

        :param func:  a callable object with interface func(ins,)
        :return:
        """
        if len(self) == 0:
            return []
        try:
            result = []
            if show_process:
                func_name = func.__name__ if type(func).__name__ == 'function' else type(func).__name__
                iterator = tqdm(enumerate(self), total=len(self), desc=func_name, ncols=100, mininterval=0.3)
            else:
                iterator = enumerate(self)
            # TODO: Multi processor to deal with big data
            for idx, item in iterator:
                result.append(func(item))
                
            
            
            if new_field_names is not None:
                assert len(new_field_names) == len(result[0]) or len(
                    new_field_names) == 1, "the length of new field names must equal to the length ofoutput"
                if len(new_field_names) == len(result[0]):
                    for result_id, new_field_name in enumerate(new_field_names):
                        new_field = [item[result_id] for item in result]
                        self.add_field(new_field_name, new_field)
                else:
                    self.add_field(new_field_names[0], result)
            return self
        
        except Exception as e:
            raise e
    
    def apply_fields(self, func, field_names, new_field_names=None):
        """

        :param func:
        :param field_names:
        :param new_field_names:
        :return:
        """
        if not isinstance(field_names, Iterable):
            field_names = [field_names]
        if new_field_names is None:
            new_field_names = [None] * len(field_names)
        elif not isinstance(new_field_names, Iterable):
            new_field_names = [new_field_names]
        assert len(field_names) == len(new_field_names), \
            "The field names must have the same length as new field names"
        assert isinstance(func, Callable), "func must be a callable instance."
        
        for field_name, new_field_name in zip(field_names, new_field_names):
            self.apply_field(func, field_name, new_field_name)
    
    def apply_field(self, func, field_name, new_field_name=None):
        """

        :param func: the callable object of func(item)
        :param field_name:
        :param new_field_name:
        :return:
        """
        idx = -1
        try:
            result = []
            for idx, item in enumerate(self[field_name]):
                result.append(func(item))
            
            if new_field_name is None:
                self.add_field(field_name, result)
            elif isinstance(new_field_name, str):
                self.add_field(new_field_name, result)
            return Field(field_name, result)
        
        except Exception as e:
            if idx != -1:
                # Todo: 1.add Log
                pass
            raise e
    
    def combine(self, combine_list: List[str],
                new_field_names,
                position_field_names=None,
                target_position_fields=None,
                new_position_field_names=None):
        """
        combine the fields into a list of str, for example

        string1 [sep] string2 [cls]
        = [field_name1 , '[SEP]', field_name2, '[CLS]']

        filed must be the list of str.

        Todo:
            1. Output the range of each field in every instance

        :param combine_list:
        :param new_field_names:
        :param position_field_names:
        :param target_position_fields:
        :param new_position_field_names:
        :return:
        """
        # check
        assert not isinstance(combine_list, str), "The combine_list only allow  the List or Tuple."
        assert isinstance(combine_list, Iterable), "The combine_list only allow  the List or Tuple."
        assert any([item in self for item in combine_list]), "combine_list must have one field name in {}".format(
            self.field_name_list)
        
        position_field_recoder = {}
        new_field_names_recoder = {}
        if position_field_names is not None:
            assert target_position_fields is not None, "position fields and origin position field cannot be None at the same time"
            if isinstance(position_field_names, str):
                position_field_names = [position_field_names]
            
            if isinstance(target_position_fields, str):
                target_position_fields = [target_position_fields]
            assert len(target_position_fields) == 1 or len(position_field_names) == len(target_position_fields), \
                "position fields must have the same length as position or length of  target_position_fields equal to 1."
            if len(target_position_fields) == 1:
                position_field_recoder[target_position_fields[0]] = position_field_names
            else:
                position_field_recoder = {k: v for k, v in zip(target_position_fields, position_field_names)}
            
            if new_position_field_names is not None:
                if isinstance(new_position_field_names, str):
                    new_position_field_names = [new_position_field_names]
                assert len(new_position_field_names) == len(position_field_names), \
                    "the new position fields must have the same length as position fields"
                new_field_names_recoder = {k: v for k, v in zip(position_field_names, new_position_field_names)}
        
        for token in combine_list:
            if token in self:
                assert self[token].dtype != str(type(list)), "Now we only allow the list to combine."
        
        # combile
        combine_result = []
        new_position_fields = defaultdict(list)
        output_position = []
        for instance in self:
            ins_list = []
            for token in combine_list:
                if token in self:
                    # deal with the position
                    if token in position_field_recoder:
                        base_len = len(ins_list)
                        for position_field_name in position_field_recoder[token]:
                            origin_postion_data = instance[position_field_name]
                            target_position_data = origin_postion_data + base_len
                            if new_field_names is not None:
                                new_field_name = new_field_names_recoder[position_field_name]
                                new_position_fields[new_field_name].append(target_position_data)
                    
                    # combine the new tokens
                    tgt_list = instance[token]
                    assert isinstance(tgt_list, list), "Cannot result the data."
                    ins_list.extend(tgt_list)
                else:
                    ins_list.append(token)
            combine_result.append(ins_list)
        
        self.add_field(new_field_names, combine_result)
        
        for field_name, field in new_position_fields.items():
            self.add_field(field_name, field)
        
        return combine_result, output_position
    
    def items(self):
        return self._data_container.items()
    
    def to_dict(self):
        return self._data_container
    
    # IO
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            result = pickle.load(f)
        if not isinstance(result, Dataset):
            raise TypeError("Load the wrong type")
        return result
    
    def save(self, save_path):
        """
        save the
        :param save_path:
        :return:
        """
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def from_SQL():
        pass
    
    def to_SQL(self):
        pass
    
    @staticmethod
    def from_txt():
        pass
    
    def to_txt(self):
        pass
    
    @staticmethod
    def from_csv():
        pass
    
    def sort_by(self, field_name=None, key=None, reverse=False, in_place=False):
        # todo: dataset sort
        if not in_place:
            dataset = copy.deepcopy(self)
        else:
            dataset = self
        if field_name is not None:
            sorted_dataset = sorted(dataset, key=lambda x: x[field_name], reverse=reverse)
        elif key is not None:
            sorted_dataset = sorted(dataset, key=key, reverse=reverse)
        else:
            raise TypeError("You must input filed name or sorted key function")
        
        return Dataset(sorted_dataset)
    
    def add_dataset(self, dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("Only accept the dataset type")
        
        assert set(self.field_name_list) == set(dataset.field_name_list), "the dataset must have the same fields"
        for item in dataset:
            self.append(item)
        
        return self
    
    def __add__(self, other):
        return self.add_dataset(other)


def test_combine():
    data = read_origin_data('../../data/train-v1.1.jsonl', limit=10)
    data = [Instance(item) for item in data]
    dataset = Dataset(data)
    tokenizer = EnglishTokenizer()
    a = []
    print(dataset)
    print(dataset[0])
    
    # res = dataset.apply_field(tokenizer, "context", "context")
    res = dataset.apply_fields(tokenizer, ["context", "question", "answer"],
                               ["token_context", "token_question", "token_answer"])
    
    print(dataset)
    
    dataset.save("dataset.pkl")
    
    loaded_dataset = Dataset.load("../../dataset.pkl")
    dataset.combine(['[BOS]', "token_context", '[SEP]', "token_question", '[CLS]'],
                    new_field_names="combine_data",
                    position_field_names=['s_idx', 'e_idx'],
                    target_position_fields=['token_context'],
                    new_position_field_names=['combined_s_idx', 'combined_e_idx'])
    print(dataset)
