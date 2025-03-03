#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2023/9/23 16:16
# @Desc  :
import os.path
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from data_set import DataSet
import utils
from utils import BPRLoss, EmbLoss
from lightGCN import LightGCN


class GraphEncoder(nn.Module):
    def __init__(
        self, device, layers, n_users, n_items, inter_matrix, dropout, dataset
    ):
        super(GraphEncoder, self).__init__()
        self.dataset = dataset
        self.light = LightGCN(device, layers, n_users, n_items, inter_matrix, dataset)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, in_embs):

        result = self.light(in_embs)
        return result


class Mutual_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, args, dataset):
        super(Mutual_Attention, self).__init__()
        # self.q = nn.Linear(input_dim, dim_qk, bias=False)
        # self.k = nn.Linear(input_dim, dim_qk, bias=False)
        # self.v = nn.Linear(dim_v, dim_v, bias=False)

        self.embedding_size = args.embedding_size
        self.behaviors = args.behaviors

    def forward(self, feature_embeddings):
        # Q = self.q(q_token)  # Q: batch_size * seq_len * dim_k
        # K = self.k(k_token)  # K: batch_size * seq_len * dim_k
        # V = self.v(v_token)  # V: batch_size * seq_len * dim_v

        # Q * K.T() # batch_size * seq_len * seq_len
        # att = nn.Softmax(dim=-1)(torch.matmul(Q, K.transpose(-1, -2)) * self._norm_fact)

        # Q * K.T() * V # batch_size * seq_len * dim_v
        # att = torch.matmul(att, V)
        attention_scores = []
        attention_scores_all = []
        attention_scores_table = []
        for i in range(len(self.behaviors)):
            behaviors_embeddinsg = feature_embeddings[:, i].unsqueeze(1)
            attention_scores_b = torch.matmul(
                behaviors_embeddinsg, feature_embeddings.transpose(-1, -2)
            )
            attention_scores_table.append(attention_scores_b)
        for i in range(len(self.behaviors) - 1):
            attention_scores_table_prj = attention_scores_table[i]
            res_array = (attention_scores_table[-1] * attention_scores_table_prj).sum(
                dim=1, keepdim=True
            ) * attention_scores_table[-1]
            norm_num = attention_scores_table[-1].norm(dim=1) ** 2 + 1e-12
            clear_emb = res_array / norm_num.unsqueeze(1)
            attention_scores.append(clear_emb)
        attention_scores_all = torch.cat(attention_scores, dim=-2)
        attention_scores = torch.cat(attention_scores, dim=-2)
        attention_scores = torch.sum(attention_scores, dim=-2).unsqueeze(1)
        attention_scores += attention_scores_table[-1]
        attention_scores_all = torch.cat(
            (attention_scores_all, attention_scores), dim=1
        )
        att = nn.Softmax(dim=-1)(attention_scores_all / sqrt(self.embedding_size))
        att = torch.matmul(att, feature_embeddings)

        return att


class ABDMBR(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(ABDMBR, self).__init__()
        self.dataset = dataset
        self.distill_userK = args.distill_userK
        self.distill_thres = args.distill_thres
        self.distill_layers = args.distill_layers
        self.testbatch = args.testbatch
        self.device = args.device
        self.layers = args.layers
        self.initializer_range = args.initializer_range
        self.reg_weight = args.reg_weight
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.inter_matrix = dataset.inter_matrix
        self.user_item_inter_set = dataset.user_item_inter_set
        self.item_behaviour_degree = dataset.item_behaviour_degree.to(self.device)
        self.test_users = list(dataset.test_interacts.keys())
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(
            self.n_users + 1, self.embedding_size, padding_idx=0
        )
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0
        )

        self.global_Graph = LightGCN(
            self.device,
            self.layers,
            self.n_users + 1,
            self.n_items + 1,
            dataset.all_inter_matrix,
            dataset,
        )

        self.Graph_encoder = nn.ModuleDict(
            {
                behavior: GraphEncoder(
                    self.device,
                    self.layers,
                    self.n_users + 1,
                    self.n_items + 1,
                    dataset.inter_matrix[behavior],
                    self.node_dropout,
                    dataset,
                )
                for behavior in self.behaviors
            }
        )

        self.W = nn.Parameter(torch.ones(len(self.behaviors)))

        self.dim_qk = args.dim_qk
        self.dim_v = args.dim_v
        self.attention = Mutual_Attention(args, dataset)

        self.reg_weight = args.reg_weight
        self.layers = args.layers
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.f = nn.Sigmoid()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        self.message_dropout = nn.Dropout(p=args.message_dropout)

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self, total_embeddings):
        """
        gcn propagate in each behavior
        """
        all_user_embeddings, all_item_embeddings = [], []
        for behavior in self.behaviors:
            # if behavior == "buy":
            #     behavior_embeddings = self.Graph_encoder[behavior](total_embeddings)
            # else :
            #     behavior_embeddings = self.Graph_encoder[behavior](total_embeddings)
            #     [distill_row, distill_col, distill_val] = self.generateKorderGraph(userK=self.distill_userK, itemK=self.distill_itemK, total_embeddings=behavior_embeddings,threshold=self.distill_thres)
            #     all_inter_matrix_pre = self.reset_graph([distill_row, distill_col, distill_val],self.inter_matrix[behavior])
            #     global_Graph_pre = LightGCN(self.device, self.distill_layers, self.n_users + 1, self.n_items + 1, all_inter_matrix_pre,self.dataset)
            #     behavior_embeddings = global_Graph_pre(total_embeddings)
            #     # behavior_embeddings = self.Graph_encoder[behavior](behavior_embeddings)
            #     # behavior_embeddings = F.normalize(behavior_embeddings, dim=-1)
            #     # all_embeddings.append(behavior_embeddings + total_embeddings)
            behavior_embeddings = self.Graph_encoder[behavior](total_embeddings)
            user_embedding, item_embedding = torch.split(
                behavior_embeddings, [self.n_users + 1, self.n_items + 1]
            )
            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)

        # target_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        # target_user_embeddings = torch.sum(target_user_embeddings, dim=1)
        # all_user_embeddings[-1] = target_user_embeddings

        all_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        all_item_embeddings = torch.stack(all_item_embeddings, dim=1)
        return all_user_embeddings, all_item_embeddings

    def reset_all(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.Graph_encoder = nn.ModuleDict(
            {
                behavior: GraphEncoder(
                    self.device,
                    self.distill_layers,
                    self.n_users + 1,
                    self.n_items + 1,
                    self.dataset.inter_matrix[behavior],
                    self.node_dropout,
                    self.dataset,
                )
                for behavior in self.behaviors
            }
        )
        self.apply(self._init_weights)

    def get_all_embeddings(self, behavior):
        all_embeddings_1 = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_embeddings = self.global_Graph(all_embeddings_1)
        behavior_embeddings = self.Graph_encoder[behavior](all_embeddings)
        return behavior_embeddings

    def forward(self, batch_data):
        all_embeddings_1 = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_embeddings = self.global_Graph(all_embeddings_1)
        # # top-k
        # [distill_row, distill_col, distill_val] = self.generateKorderGraph(userK=self.distill_userK, itemK=self.distill_itemK, total_embeddings=all_embeddings,threshold=self.distill_thres)
        # all_inter_matrix_pre = self.reset_graph([distill_row, distill_col, distill_val],self.n_users,self.n_items)
        # global_Graph_pre = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, all_inter_matrix_pre,self.dataset)
        # all_embeddings_pre = global_Graph_pre(all_embeddings_1)

        # all_embeddings_pre = all_embeddings_pre + all_embeddings
        user_embedding, item_embedding = torch.split(
            all_embeddings, [self.n_users + 1, self.n_items + 1]
        )

        all_user_embeddings, all_item_embeddings = self.gcn_propagate(all_embeddings)
        # all_item_embeddings_now=torch.sum(all_item_embeddings, dim=1)
        all_user_embeddings = self.attention(all_user_embeddings)
        # all_user_embeddings = all_user_embeddings + user_embedding.unsqueeze(1)
        # all_item_embeddings = self.attention(all_item_embeddings)
        # all_item_embeddings = all_item_embeddings.squeeze(1)
        weight = self.item_behaviour_degree * self.W
        weight = weight / (torch.sum(weight, dim=1).unsqueeze(-1) + 1e-8)
        all_item_embeddings = all_item_embeddings * weight.unsqueeze(-1)
        # # all_item_embeddings = torch.sum(all_item_embeddings, dim=1) + item_embedding
        all_item_embeddings = torch.sum(all_item_embeddings, dim=1)
        total_loss = 0
        total_loss1 = 0

        for i in range(len(self.behaviors)):
            data = batch_data[:, i]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            user_feature = all_user_embeddings[:, i][users.view(-1, 1)]
            item_feature = all_item_embeddings[items]
            # user_feature, item_feature = self.message_dropout(user_feature), self.message_dropout(item_feature)
            scores = torch.sum(user_feature * item_feature, dim=2)
            total_loss1 += self.bpr_loss(scores[:, 0], scores[:, 1])
        # for i,behavior in enumerate(self.behaviors):
        #     if behavior == 'buy':
        #         data = batch_data[:, i]
        #         users = data[:, 0].long()
        #         items = data[:, 1:].long()
        #         user_feature = all_user_embeddings_attn[:, i][users.view(-1, 1)]
        #         item_feature = all_item_embeddings_w[items]
        #         # user_feature, item_feature = self.message_dropout(user_feature), self.message_dropout(item_feature)
        #         scores_attn = torch.sum(user_feature * item_feature, dim=2)
        #         total_loss2 += self.bpr_loss(scores_attn[:, 0], scores_attn[:, 1])
        total_loss = total_loss1 + self.reg_weight * self.emb_loss(
            self.user_embedding.weight, self.item_embedding.weight
        )

        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            all_embeddings_1 = torch.cat(
                [self.user_embedding.weight, self.item_embedding.weight], dim=0
            )
            all_embeddings = self.global_Graph(all_embeddings_1)
            # top-k
            # [distill_row, distill_col, distill_val] = self.generateKorderGraph(userK=self.distill_userK, itemK=self.distill_itemK, total_embeddings=all_embeddings,threshold=self.distill_thres)
            # all_inter_matrix_pre = self.reset_graph([distill_row, distill_col, distill_val],self.n_users,self.n_items)
            # global_Graph_pre = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, all_inter_matrix_pre,self.dataset)
            # all_embeddings_pre = global_Graph_pre(all_embeddings_1)
            # all_embeddings_pre = all_embeddings_pre + all_embeddings

            user_embedding, item_embedding = torch.split(
                all_embeddings, [self.n_users + 1, self.n_items + 1]
            )
            all_user_embeddings, all_item_embeddings = self.gcn_propagate(
                all_embeddings
            )

            # target_embeddings = all_user_embeddings[:, -1].unsqueeze(1)

            target_embeddings = self.attention(all_user_embeddings)

            all_user_embeddings = self.attention(all_user_embeddings)
            target_embeddings = all_user_embeddings[:, -1].unsqueeze(1)
            self.storage_user_embeddings = target_embeddings.squeeze()
            # all_item_embeddings = self.attention(all_item_embeddings)
            weight = self.item_behaviour_degree * self.W
            weight = weight / (torch.sum(weight, dim=1).unsqueeze(-1) + 1e-8)
            all_item_embeddings = all_item_embeddings * weight.unsqueeze(-1)
            self.storage_item_embeddings = torch.sum(all_item_embeddings, dim=1)

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores
