#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import random
import numpy as np
import torch
import sys
import time
import tempfile
import shutil
import os
import torch.nn as nn
import pickle as pkl
import pandas as pd
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append("..") 

from torch.utils.data.dataloader import DataLoader
from utils.options import args_parser
from utils.sampling import Argoverse_iid, Argoverse_noniid
#from utils.lstm_utils import ModelUtils, LSTMDataset, EncoderRNN, DecoderRNN, train, validate, evaluate, infer_helper, get_city_names_from_features, get_m_trajectories_along_n_cl, get_pruned_guesses, viz_predictions_helper
from utils.update import LocalUpdate, DatasetSplit
from utils.federate_learning_avg import FedAvg
from utils.plot_utils import min_ignore_None, plot_loss_acc_curve
from utils.log_utils import save_training_log
from typing import Any, Dict, List, Tuple, Union
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from utils.logger import Logger

from importlib import import_module

import torch
from torch.utils.data import DataLoader


from utils.utils import Logger, load_pretrain, save_ckpt, evaluate

def CL_training(args,city):
    # parse args

    # fix random_seed, so that the experiment can be repeat
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f"Using device ({args.device}) ...")
    #model_utils = ModelUtils()

    # build model
    if args.model in ['lanegcn', 'LaneGCN']:
        model = import_module(args.model)
        config, Dataset, collate_fn, net, Loss, post_process = model.get_model(args)

        #load trained model
        if args.resume or args.weight:
            ckpt_path = args.resume or args.weight
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(config["save_dir"], ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=args.device)
            load_pretrain(net, ckpt["state_dict"])
            if args.resume:
                config["epoch"] = ckpt["epoch"]
        net_glob = copy.deepcopy(net)
        print(net_glob)
    else:
        exit('Error: unrecognized model')

    # load dataset and split users
    num_users = 1
    if args.dataset == 'Argoverse':
        # Get PyTorch Dataset
        print("Loading dataset")
        start_time = time.time()
        dataset_train = Dataset(args.train_features, config, train=True)
        dataset_val = Dataset(args.val_features, config, train=False)

        # non-i.i.d. dataset by city name
        train_city_dict = {'MIA':[], 'PIT':[]}
        val_city_dict = {'MIA':[], 'PIT':[]}
        
        for idx,data in enumerate(dataset_train.split):
            train_city_dict[data['city']].append(idx)
        for idx,data in enumerate(dataset_val.split):
            val_city_dict[data['city']].append(idx)

        args.local_iter = int(len(train_city_dict[city])/args.local_bs)

        end_time = time.time()
        print("Complete dataset loading with running time {:.3f}s".format(end_time-start_time))
        print("MIA:{}, PIT:{}".format(len(train_city_dict["MIA"]),len(train_city_dict["PIT"])))

    else:
        exit('Error: unrecognized dataset')
    
    # Create log and copy all code
    save_dir = config["save_dir"]
    log_save_dir = os.path.join(save_dir, "log")
    ckpt_save_dir = os.path.join(save_dir, "ckpt")
    log_save_path = os.path.join(log_save_dir, args.save_address_id)
    ckpt_save_path = os.path.join(ckpt_save_dir, args.save_address_id)
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)
    sys.stdout = Logger(log_save_path)

    net_glob.train()
    
    # training
    train_loss_list = []
    val_same_loss_list = []
    val_other_loss_list = []
    eval_same_metrices_list = []
    eval_other_metrices_list = []
    #net_best = net_glob
    rounds = 100
    for round in range(rounds):
        print("Round {:3d} Training start".format(round))
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=train_city_dict[city], local_bs=args.local_bs)
        w_glob, loss = local.train(net=copy.deepcopy(net_glob), config=config, local_iter=args.local_iter)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = copy.deepcopy(loss)
        print('Round {:3d}, Average Training Loss {:.5f}'.format(round, loss_avg))
        train_loss_list.append(loss_avg)

        # save checkpoint
        print('save ckpt')
        save_ckpt(net_glob, ckpt_save_path, round)

        # validation part
        val_loader_same_city = DataLoader(
            DatasetSplit(dataset_val, val_city_dict[city]),
            batch_size=args.local_bs,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        other_city = 'PIT' if city=='MIA' else 'MIA'
        val_loader_other_city = DataLoader(
            DatasetSplit(dataset_val, val_city_dict[other_city]),
            batch_size=args.local_bs,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print('val begin')

        round_val_loss, _cls, _reg, ade1, fde1, mr1, ade, fde, mr = val(args, val_loader_same_city, net_glob, Loss, post_process, round)
        val_same_loss_list.append(round_val_loss)
        metric_results = {"minADE": ade, "minFDE": fde, "MR": mr, "minADE1": ade1, "minFDE1": fde1, "MR1": mr1, "DAC": None}
        print('Same city metric_results:{}'.format(metric_results))
        eval_same_metrices_list.append(metric_results)
        plot_loss_acc_curve(args, train_loss_list, val_same_loss_list, eval_same_metrices_list, rounds)
        save_training_log(args, train_loss_list, val_same_loss_list, eval_same_metrices_list)

        round_val_loss, _cls, _reg, ade1, fde1, mr1, ade, fde, mr = val(args, val_loader_other_city, net_glob, Loss, post_process, round)
        val_other_loss_list.append(round_val_loss)
        metric_results = {"minADE": ade, "minFDE": fde, "MR": mr, "minADE1": ade1, "minFDE1": fde1, "MR1": mr1, "DAC": None}
        print('Other city metric_results:{}'.format(metric_results))
        eval_other_metrices_list.append(metric_results)


        #plot_loss_acc_curve(args, train_loss_list, val_other_loss_list, eval_other_metrices_list, rounds)
        #save_training_log(args, train_loss_list, val_other_loss_list, eval_other_metrices_list)




def val(args, data_loader, net, loss, post_process, epoch):
    net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        #print('\r val progress: {}/{}'.format(i,len(data_loader)), end="")
        with torch.no_grad():
            output = net(data)
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    loss, cls, reg, ade1, fde1, mr1, ade, fde, mr = post_process.display(metrics, dt, epoch)

    net.train()
    return loss, cls, reg, ade1, fde1, mr1, ade, fde, mr 