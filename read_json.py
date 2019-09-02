#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:36:17 2019

@author: robot
"""

import argparse
import json
import shutil
import os
from bunch import Bunch

json_path = './read_param.json'

def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    create dir
    :param dir_name: 文件夹列表
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(u'[INFO] Dir "%s" exists, deleting.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(u'[INFO] Dir "%s" not exists, creating.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False

def get_config_from_json(json_file):
    """
    将配置文件转换为配置类
    change json file to dictionary
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  # 配置字典

    config = Bunch(config_dict)  # 将配置字典转换为类
    print('config well')
    return config, config_dict

config, _ = get_config_from_json(json_path)

