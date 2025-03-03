#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author:
# @Date  : 2023/9/23 15:25
# @Desc  :
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from data_set import DataSet
from model_pre_1_prj_m import ABDMBR

from trainer_pre import Trainer


seed = 5754
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # True can improve train speed
    torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Set args", add_help=False)

    parser.add_argument("--embedding_size", type=int, default=64, help="")
    parser.add_argument("--reg_weight", type=float, default=1e-3, help="")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--node_dropout", type=float, default=0.75)
    parser.add_argument("--message_dropout", type=float, default=0.25)
    parser.add_argument("--dim_qk", type=int, default=32)
    parser.add_argument("--dim_v", type=int, default=64)
    parser.add_argument("--omega", type=float, default=1)
    parser.add_argument("--data_name", type=str, default="beibei", help="")
    parser.add_argument("--behaviors", help="", action="append")
    parser.add_argument("--loss_type", type=str, default="bpr", help="")
    parser.add_argument("--if_load_model", type=bool, default=False, help="")
    parser.add_argument("--gpu_no", type=int, default=1, help="")
    parser.add_argument("--topk", type=list, default=[1, 5, 10, 15, 20, 40], help="")
    parser.add_argument("--metrics", type=list, default=["hit", "ndcg"], help="")
    parser.add_argument("--lr", type=float, default=0.001, help="")
    parser.add_argument("--decay", type=float, default=0.001, help="")
    parser.add_argument("--batch_size", type=int, default=1024, help="")
    parser.add_argument("--test_batch_size", type=int, default=1024, help="")
    parser.add_argument("--min_epoch", type=str, default=5, help="")
    parser.add_argument("--epochs", type=str, default=100, help="")
    parser.add_argument("--model_path", type=str, default="./check_point", help="")
    parser.add_argument("--check_point", type=str, default="", help="")
    parser.add_argument("--model_name", type=str, default="model_pre_1", help="")
    parser.add_argument("--pt_loop", type=int, default=50, help="")
    parser.add_argument("--device", type=str, default="cuda:1", help="")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="")
    # #top_k
    # #distill K and threshold
    # parser.add_argument("--distill_userK", type=int, default=3, help="distill K")
    # parser.add_argument("--distill_uuK", type=int, default=5, help="distill K")
    # parser.add_argument("--distill_itemK", type=int, default=3, help="distill K")
    # parser.add_argument("--distill_iiK", type=int, default=5, help="distill K")
    # parser.add_argument("--distill_layers", type=int, default=2, help="distill number of layers")
    # parser.add_argument("--distill_thres", type=float, default=0.5, help="distill threshold")
    # parser.add_argument("--uuii_thres", type=float, default=-1, help="distill threshold")
    # parser.add_argument("--uu_lambda", type=float, default=100, help="lambda for ease")
    # parser.add_argument("--ii_lambda", type=float, default=200 , help="lambda for ease")
    # parser.add_argument('--testbatch', type=int,default=1024,help="the batch size of users for testing")

    args = parser.parse_args()
    if args.data_name == "tmall":
        args.data_path = "./data/Tmall"
        args.behaviors = ["click", "collect", "cart", "buy"]
    elif args.data_name == "beibei":
        args.data_path = "./data/beibei"
        args.behaviors = ["view", "cart", "buy"]
    elif args.data_name == "taobao":
        args.data_path = "./data/taobao"
        args.behaviors = ["view", "cart", "buy"]
    else:
        raise Exception("data_name cannot be None")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME

    logfile = "{}_enb_{}_{}_{}".format(
        args.data_name, args.embedding_size, args.lr, TIME
    )
    # args.train_writer = SummaryWriter('./log/train/' + logfile)
    # args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add("./log/{}/{}.log".format(args.model_name, logfile), encoding="utf-8")

    start = time.time()
    dataset = DataSet(args)
    model = ABDMBR(args, dataset).to(args.device)

    trainer = Trainer(model, dataset, args)

    logger.info(args.__str__())
    logger.info(model)

    trainer.train_model()
    # trainer.evaluate(0, 5, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, args.test_writer)
    logger.info("train end total cost time: {}".format(time.time() - start))
