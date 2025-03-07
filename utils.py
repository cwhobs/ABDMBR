#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author:
# @Date  : 2023/9/23 11:26
# @Desc  :
import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = 0
        for embedding in embeddings:
            tmp = torch.norm(embedding, p=self.norm)
            tmp = tmp / embedding.shape[0]
            emb_loss += tmp
        return emb_loss

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 256)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)