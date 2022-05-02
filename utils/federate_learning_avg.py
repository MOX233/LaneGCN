#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
    
def FedAvg_weighted(w, c, skew=0.5):
    eps = 1e-12
    w_avg = copy.deepcopy(w[0])
    w_avg = skew * w_avg if c[0]=='MIA' else  (1 - skew) * w_avg
    div_norm = 0 + eps
    for city in c:
        div_norm += skew if city=='MIA' else  (1 - skew)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += skew * w[i][k] if c[i]=='MIA' else  (1 - skew) * w[i][k]
        w_avg[k] = torch.div(w_avg[k], div_norm)
    return w_avg
