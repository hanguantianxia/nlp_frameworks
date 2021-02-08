#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            git_tool.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/9/14 11:18    
@Version         1.0 
@Desciption 

'''

import git
from git import Repo


class GitManager():
    def __init__(self, repo: git.Repo):
        self.repo = repo
    
    def _get_commit_mess(self):
        head = self.repo.head
        
        last_commit = head.commit
        commit_mess = last_commit.message
        commit_hexsha = last_commit.hexsha
        
        return commit_mess, commit_hexsha
    
    def get_commit_mess(self, str_format=None):
        commit_mess, commit_hexsha = self._get_commit_mess()
        if str_format is None:
            return str(commit_mess) + "\t" + str(commit_hexsha)
    
    @classmethod
    def get_repo(cls, direction='.'):
        """
        get repo
        :param direction:
        :return:
        """
        repo = Repo(direction)
        
        return cls(repo)


if __name__ == '__main__':
    git_manager = GitManager.get_repo(".")
