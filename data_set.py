#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_set.py
# @Author:
# @Date  : 2021/11/1 11:38
# @Desc  :
import argparse
import os
from os.path import join
import sys
import random
import json
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import copy
import pickle

# SEED = 2378
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class BehaviorDate(Dataset):
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []
        all_inter = self.behavior_dict['all'].get(str(idx + 1), None)
        for behavior in self.behaviors:

            items = self.behavior_dict[behavior].get(str(idx + 1), None)
            if items is None:
                signal = [0, 0, 0]
            else:
                pos = random.sample(items, 1)[0]
                neg = random.randint(1, self.item_count)
                while np.isin(neg, all_inter):
                    neg = random.randint(1, self.item_count)
                signal = [idx + 1, pos, neg]
            total.append(signal)

        return np.array(total)

    def __len__(self):
        return self.user_count



class DataSet(object):

    def __init__(self, args):

        self.behaviors = args.behaviors
        self.path = args.data_path
        self.loss_type = args.loss_type

        self.__get_count()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()
        self.__get_trainUser()
        # self.__get_behavior_items
        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_behavior_items(self):
        """
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict
          
    def __get_trainUser(self):
        self.trainUniqueUsers = []
        self.trainItem=[]
        self.trainUser=[]
        lines = open(os.path.join(self.path, 'all1.txt'), encoding='utf-8').readlines()
        for line in lines:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            uid = int(user) 
            itemids = [int(itemstr) for itemstr in items]
            self.trainUniqueUsers.append(uid)
            self.trainUser.extend([uid] * (len(items)))
            self.trainItem.extend(itemids)
    
    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        """
        load the list of items that the user has interacted with in the validation set
        :return:
        """
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def __get_sparse_interact_dict(self):
        """
        load graphs

        :return:
        """
        # self.edge_index = {}
        self.item_behaviour_degree = []
        self.user_behaviour_degree = []
        # self.all_item_user = {}
        # self.behavior_item_user = {}
        self.inter_matrix = {}
        self.user_item_inter_set = []
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            # tmp_dict = {}
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))

                    # if line[1] in self.all_item_user:
                    #     self.all_item_user[line[1]].append(int(line[0]))
                    # else:
                    #     self.all_item_user[line[1]] = [int(line[0])]

                    # if line[1] in tmp_dict:
                    #     tmp_dict[line[1]].append(int(line[0]))
                    # else:
                    #     tmp_dict[line[1]] = [int(line[0])]
                # self.behavior_item_user[behavior] = tmp_dict
                indices = np.vstack((row, col))
                indices = torch.LongTensor(indices)

                values = torch.ones(len(row), dtype=torch.float32)
                inter_matrix = sp.coo_matrix((values, (row, col)), [self.user_count + 1, self.item_count + 1])
                user_item_set = [list(row.nonzero()[1]) for row in inter_matrix.tocsr()]
                self.inter_matrix[behavior]=inter_matrix
                self.user_item_inter_set.append(user_item_set)

                # user_item_set = [set(row.nonzero()[1]) for row in user_item_inter_matrix]
                # item_user_set = [set(user_inter[:, j].nonzero()[0]) for j in range(user_inter.shape[1])]

                user_inter = torch.sparse.FloatTensor(indices, values, [self.user_count + 1, self.item_count + 1]).to_dense()
                self.item_behaviour_degree.append(user_inter.sum(dim=0))
                self.user_behaviour_degree.append(user_inter.sum(dim=1))
                # col = [x + self.user_count + 1 for x in col]
                # row, col = [row, col], [col, row]
                # row = torch.LongTensor(row).view(-1)
                all_row.extend(row)
                # col = torch.LongTensor(col).view(-1)
                all_col.extend(col)
                # edge_index = torch.stack([row, col])
                # self.edge_index[behavior] = edge_index
        # self.all_item_user = {key: list(set(value)) for key, value in self.all_item_user.items()}
        self.item_behaviour_degree = torch.stack(self.item_behaviour_degree, dim=0).T
        self.user_behaviour_degree = torch.stack(self.user_behaviour_degree, dim=0).T
        # all_row = torch.cat(all_row, dim=-1)
        # all_col = torch.cat(all_col, dim=-1)
        # all_row = all_row.tolist()
        # all_col = all_col.tolist()
        # self.all_edge_index = list(set(zip(all_row, all_col)))
        all_edge_index = list(set(zip(all_row, all_col)))
        all_row = [sub[0] for sub in all_edge_index]
        all_col = [sub[1] for sub in all_edge_index]
        values = torch.ones(len(all_row), dtype=torch.float32)
        self.all_inter_matrix = sp.coo_matrix((values, (all_row, all_col)), [self.user_count + 1, self.item_count + 1])

        # self.all_edge_index = torch.LongTensor(self.all_edge_index).T

    def reset_graph(self,newdata,user_count,item_count, behavior):
        new_row, new_col, new_val = newdata
        inter_matrix_c = self.inter_matrix[behavior]
        inter_matrix_c.data = np.hstack([inter_matrix_c.data,new_val])
        inter_matrix_c.row = np.hstack([inter_matrix_c.row,new_row])
        inter_matrix_c.col = np.hstack([inter_matrix_c.col,new_col])
        self.inter_matrix[behavior]=inter_matrix_c
        
    
    def behavior_dataset(self):
        return BehaviorDate(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors)


    def validate_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['cart', 'click', 'collect', 'buy'], help='')
    parser.add_argument('--data_path', type=str, default='./data/Tmall', help='')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    args = parser.parse_args()
    dataset = DataSet(args)
    loader = DataLoader(dataset=dataset.behavior_dataset(), batch_size=5)
    for index, item in enumerate(loader):
        print(index, '-----', item)
