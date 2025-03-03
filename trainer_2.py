#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : trainer.py
# @Author:
# @Date  : 2021/11/1 15:45
# @Desc  :
import copy
import time
import os

import torch
import numpy as np
import torch.nn as nn
import utils
import torch.nn.functional as F
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from data_set import DataSet
from metrics import metrics_dict


class Trainer(object):

    def __init__(self, model, dataset: DataSet, args):
        self.model = model
        self.dataset = dataset
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.testbatch = args.testbatch
        self.f = nn.Sigmoid()
        self.embedding_size = args.embedding_size
        self.behaviors = args.behaviors
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.min_epoch = args.min_epoch
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name

        self.device = args.device
        self.TIME = args.TIME

        self.optimizer = self.get_optimizer(self.model)

    def get_optimizer(self, model):
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def clear_parameter(self, model):
        # for device in model.device_ids:
        #     model.module.storage_user_embeddings = None
        #     model.module.storage_item_embeddings = None
        model.storage_user_embeddings = None
        model.storage_item_embeddings = None

    def generateKorderGraph(self, userK, behavior, topk, threshold=0.5):
        u_batch_size = self.testbatch
        pred_list = None

        assert userK <= 50

        distill_user_row = []
        distill_item_col = []
        distill_value = []

        behavior_embeddings = self.model.get_all_embeddings(behavior)
        items = self.dataset.train_behavior_dict[behavior]
        if userK > 0:
            with torch.no_grad():
                users = [i for i in range(self.dataset.user_count + 1)]
                user_embedding, item_embedding = torch.split(
                    behavior_embeddings, [self.n_users + 1, self.n_items + 1]
                )
                rating_pred = self.f(torch.matmul(user_embedding, item_embedding.t()))
                for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                    ##从大到小的索引（取10个）
                    rating_pred_t = rating_pred[batch_users]
                    # rating_pred_m = rating_pred_m.cpu().data.numpy().copy()
                    # ind = np.argpartition(rating_pred_m, -50)[:, -50:]
                    # arr_ind = rating_pred_m[np.arange(len(rating_pred_m))[:, None], ind]
                    # arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred_m)), ::-1]
                    # batch_pred_list = ind[np.arange(len(rating_pred_m))[:, None], arr_ind_argsort]
                    # ##取到userK个
                    # partial_batch_pred_list = batch_pred_list[:, :userK]
                    sorted_values, sorted_indices = torch.sort(
                        rating_pred_t, descending=True
                    )
                    partial_batch_pred_list_in = sorted_indices[:, :userK]
                    # partial_batch_pred_list_out = sorted_indices[:, (-userK):]
                    for batch_i in range(partial_batch_pred_list_in.shape[0]):
                        if batch_i == 0:
                            continue
                        uid = batch_users[batch_i]
                        if str(uid) in items.keys():
                            user_pred = partial_batch_pred_list_in[batch_i]
                            for eachpred in user_pred:
                                if eachpred in items[str(uid)]:
                                    continue
                                else:
                                    distill_user_row.append(uid)
                                    distill_item_col.append(int(eachpred))
                                    pred_val = rating_pred_t[batch_i, eachpred]

                                    if pred_val > threshold:
                                        distill_value.append(1)
                                    else:
                                        distill_value.append(0)
                        else:
                            continue

        return [distill_user_row, distill_item_col, distill_value]

    @logger.catch()
    def train_model(self):
        train_dataset_loader = DataLoader(
            dataset=self.dataset.behavior_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # best_result = np.zeros(len(self.topk) * len(self.metrics))
        best_result_t = 0
        best_result_v = 0
        best_dict = {}
        best_epoch = 0
        best_model = None
        final_test = None
        for epoch in range(self.epochs):

            self.model.train()
            test_metric_dict, validate_metric_dict = self._train_one_epoch(
                train_dataset_loader, epoch
            )

            if test_metric_dict is not None:
                result_t = test_metric_dict["hit@10"]
                result_v = validate_metric_dict["hit@10"]
                # early stop
                if result_t - best_result_t > 0 or result_v - best_result_v > 0:
                    if result_t - best_result_t > 0:
                        best_result_t = result_t
                        final_test = test_metric_dict
                        best_dict = validate_metric_dict
                        best_model = copy.deepcopy(self.model)
                    if result_v - best_result_v > 0:
                        best_result_v = result_v
                    best_epoch = epoch
                if epoch - best_epoch > 4:
                    break
        # save the best model
        self.save_model(best_model)
        logger.info(
            f"training end, best iteration %d, results: %s"
            % (best_epoch + 1, best_dict.__str__())
        )

        logger.info(f"final test result is:  %s" % final_test.__str__())

    def _train_one_epoch(self, behavior_dataset_loader, epoch):
        start_time = time.time()
        behavior_dataset_iter = tqdm(
            enumerate(behavior_dataset_loader),
            total=len(behavior_dataset_loader),
            desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m",
        )
        total_loss = 0.0
        batch_no = 0
        for batch_index, batch_data in behavior_dataset_iter:
            start = time.time()
            batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(batch_data)
            # loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            batch_no = batch_index + 1
            total_loss += loss.item()

        total_loss = total_loss / batch_no

        epoch_time = time.time() - start_time
        logger.info(
            "epoch %d %.2fs Train loss is [%.4f] " % (epoch + 1, epoch_time, total_loss)
        )

        self.clear_parameter(self.model)
        # validate
        start_time = time.time()
        validate_metric_dict = self.evaluate(
            epoch,
            self.test_batch_size,
            self.dataset.validate_dataset(),
            self.dataset.validation_interacts,
            self.dataset.validation_gt_length,
        )
        epoch_time = time.time() - start_time
        logger.info(
            f"validate %d cost time %.2fs, result: %s "
            % (epoch + 1, epoch_time, validate_metric_dict.__str__())
        )

        # test
        start_time = time.time()
        test_metric_dict = self.evaluate(
            epoch,
            self.test_batch_size,
            self.dataset.test_dataset(),
            self.dataset.test_interacts,
            self.dataset.test_gt_length,
        )
        epoch_time = time.time() - start_time
        logger.info(
            f"test %d cost time %.2fs, result: %s "
            % (epoch + 1, epoch_time, test_metric_dict.__str__())
        )

        return test_metric_dict, validate_metric_dict

    @logger.catch()
    @torch.no_grad()
    def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length):
        data_loader = DataLoader(dataset=dataset, batch_size=test_batch_size)
        self.model.eval()
        start_time = time.time()
        iter_data = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f"\033[1;35mEvaluate \033[0m",
        )
        topk_list = []
        train_items = self.dataset.train_behavior_dict[self.behaviors[-1]]
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(self.device)
            start = time.time()
            # scores = self.model.module.full_predict(batch_data)
            scores = self.model.full_predict(batch_data)

            for index, user in enumerate(batch_data):
                user_score = scores[index]
                items = train_items.get(str(user.item()), None)
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                gt_items = gt_interacts[str(user.item())]
                mask = np.isin(topk_idx.to("cpu"), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        return metric_dict

    def calculate_result(self, topk_list, gt_len):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                key = "{}@{}".format(metric, topk)
                metric_dict[key] = np.round(value[topk - 1], 4)

        return metric_dict

    def save_model(self, model):
        torch.save(
            model.state_dict(),
            os.path.join(self.model_path, self.model_name + "_" + self.TIME + ".pth"),
        )
