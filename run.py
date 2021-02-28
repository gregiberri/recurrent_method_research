#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 22:54
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : run.py
"""

import argparse
import logging
import warnings

from ml.solver import Solver

warnings.filterwarnings("ignore")
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--mode', type=str, default='train', choices=['visualize_autoencode', 'pretrain',
                                                                  'train', 'val', 'resume', 'hyperopt'],
                    help='The mode of the running.')
parser.add_argument('-c', '--config', type=str, default='config/config_files/irmas_all.yaml', help='Config file name')

args = parser.parse_args()
solver = Solver(args)
solver.run()

