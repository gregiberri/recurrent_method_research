# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 下午5:09
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : base_dataset.py

import torch
import numpy as np

import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, config, is_train):
        super(BaseDataset, self).__init__()
        self.config = config
        self.is_train = is_train
        self.dataset_path = self.config.path
        self.split = self.config.split
        self.split = self.split[0] if is_train else self.split[1]

    def _get_filepaths(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.filenames['image_paths'])

    def __getitem__(self, index):
        raise NotImplementedError()

