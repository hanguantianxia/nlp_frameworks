#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            logger.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/8/26 11:01    
@Version         1.0 
@Desciption 

'''

import logging
import os
import sys
import time
from typing import List


def singleton(cls):
    instances = {}
    
    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return _singleton


@singleton
class Logger():
    def __init__(self, logfile=None):
        self.logger = logging.getLogger()
        formater = logging.Formatter('%(asctime)s %(name)s  %(levelname)s %(filename)s  %(lineno)d '
                                     '%(thread)d %(threadName)s %(process)d %(message)s')
        if logfile == None:
            cur_path = os.path.split(os.path.realpath(__file__))[0]
            stime = time.strftime("%Y-%m-%d", time.localtime())
            logfile = cur_path + os.sep + "log_" + stime + ".log"
        else:
            logfile = logfile
        self.sh = logging.StreamHandler(sys.stdout)
        self.sh.setFormatter(formater)
        self.fh = logging.FileHandler(logfile)
        self.fh.setFormatter(formater)
        self.logger.addHandler(self.sh)
        self.logger.addHandler(self.fh)
        self.logger.setLevel(logging.WARNING)


class LoggerManager:
    
    def __init__(self):
        self._loggers = {}
    
    def add_logger(self, logger: logging.Logger):
        self._loggers[logger.name] = logger
    
    def remove_logger(self, logger_name):
        try:
            del self._loggers[logger_name]
        except KeyError:
            print("Cannot Delete the logger")
    
    def get_logger(self, logger_name, logger_level, logger_file, handlers: List[logging.Handler] = None,
                   formatter=None):
        if logger_name in self._loggers:
            return self[logger_name]
        logger = LoggerManager.set_logger(logger_name, logger_level, logger_file, handlers,
                                          formatter)
        
        self.add_logger(logger)
        return logger
    
    def __contains__(self, item):
        return item in self._loggers
    
    @classmethod
    def set_logger(cls, logger_name, logger_level, logger_file, handlers: List[logging.Handler] = None,
                   formatter=None):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logger_level)
        
        if formatter is None:
            formatter = logging.Formatter("[%(asctime)s-%(name)s] %(levelname)s: %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
        
        if handlers is None:
            sh = LoggerManager.get_stream_handler(logger_level, formatter)
            fh = LoggerManager.get_file_handler(logger_level, logger_file, formatter)
            logger.addHandler(sh)
            logger.addHandler(fh)
        else:
            for handler in handlers:
                logger.addHandler(handler)
                logger.setLevel(logger_level)
        return logger
    
    @staticmethod
    def get_stream_handler(handler_level, formatter=None):
        handler = logging.StreamHandler()
        handler.setLevel(handler_level)
        if formatter is None:
            formatter = logging.Formatter("[%(asctime)s-%(name)s] %(levelname)s: %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        
        return handler
    
    @staticmethod
    def get_file_handler(handler_level, handle_file, formatter=None):
        handler = logging.FileHandler(handle_file)
        handler.setLevel(handler_level)
        if formatter is None:
            formatter = logging.Formatter("[%(asctime)s-%(name)s] %(levelname)s: %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        return handler
    
    @staticmethod
    def get_formatter(format_str="[%(asctime)s-%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"):
        formatter = logging.Formatter(format_str, datefmt=datefmt)
        return formatter
    
    def __getitem__(self, item):
        return self._loggers[item]
    
    def __setitem__(self, key, value):
        self._loggers[key] = value


if __name__ == '__main__':
    logger_manager = LoggerManager()
    
    logger = logger_manager.get_logger("test", logging.INFO, "test.log")
