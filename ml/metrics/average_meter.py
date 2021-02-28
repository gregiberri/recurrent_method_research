# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 下午4:09
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : average_meter.py
import collections
import copy
import csv
import os

import numpy as np
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, save_dir, tag):
        self.save_dir = save_dir
        self.tag = tag
        self.reset()
        self.make_metrics_file()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        self.count = 1 if self.count == 0 else self.count
        return self.sum / self.count

    def values(self):
        return self.val

    def make_metrics_file(self):
        file_path = os.path.join(self.save_dir, f'{self.tag}_results.csv')
        if not os.path.exists(file_path):
            with open(file_path, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'loss'])
                writer.writeheader()

    def save_metrics(self, epoch):
        # make result dict
        with open(os.path.join(self.save_dir, f'{self.tag}_results.csv'), mode='a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'loss'])
            writer.writerow({'epoch': epoch, 'loss': self.mean()})
