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
from utils.update import LocalUpdate    # need to rewrite
from utils.federate_learning_avg import FedAvg
from utils.plot_utils import min_ignore_None, plot_loss_acc_curve
from utils.log_utils import save_training_log
from typing import Any, Dict, List, Tuple, Union
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
from utils.logger import Logger

from importlib import import_module
from numbers import Number

import torch
from torch.utils.data import Sampler, DataLoader


from utils.utils import Logger, load_pretrain

def FL_training(args,FL_table,car_tripinfo):
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
        config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()

        #load trained model
        if args.resume or args.weight:
            ckpt_path = args.resume or args.weight
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(config["save_dir"], ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            load_pretrain(net, ckpt["state_dict"])
            if args.resume:
                config["epoch"] = ckpt["epoch"]
        net_glob = copy.deepcopy(net)
        print(net_glob)
    else:
        exit('Error: unrecognized model')

    # load dataset and split users
    num_users = len(car_tripinfo)
    if args.dataset == 'Argoverse':
        # Get PyTorch Dataset
        print("Preparing dataset")
        start_time = time.time()
        dataset_train = Dataset(args.train_features, config)
        dataset_val = Dataset(args.val_features, config)

        if not args.non_iid:
            dict_users = Argoverse_iid(dataset_train, args.num_items, num_users)
        else:
            # TODO: non-i.i.d. dataset
            exit('We have not realized Argoverse_noniid')
        end_time = time.time()
        print("Complete dataset preparation with running time {}s".format(end_time-start_time))

    else:
        exit('Error: unrecognized dataset')
    
    # Create log and copy all code
    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sys.stdout = Logger(log)

    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()

    # training
    train_loss_list = []
    val_loss_list = []
    eval_metrices_list = []
    #net_best = net_glob
    rounds = len(FL_table.keys())
    for round in range(rounds):
        print("Round {:3d} Training start".format(round))
        loss_locals = []
        w_locals = []
        idxs_users = [int(car.split('_')[-1]) for car in FL_table[round].keys()]
        if idxs_users == []:

            # print loss
            loss_avg = train_loss_list[-1] if round>0 else None
            if loss_avg==None:
                print('Round {:3d}, No Car, Average Training Loss None'.format(round))
            else:
                print('Round {:3d}, No Car, Average Training Loss {:.5f}'.format(round, loss_avg))
            train_loss_list.append(loss_avg)
    
            # validation part
            round_val_loss = val_loss_list[-1] if round>0 else None
            metric_results = eval_metrices_list[-1] if round>0 else {"minADE": None, "minFDE": None, "MR": None, "DAC": None}
            val_loss_list.append(round_val_loss)
            eval_metrices_list.append(metric_results)
            print("Validation Metrices: {}".format(metric_results))
            
        else:
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], local_bs=args.local_bs)
                print("localUpdate start for user {}".format(idx))
                w, loss = local.train(net=copy.deepcopy(net_glob), config=config, local_iter=args.local_iter)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = FedAvg(w_locals)
    
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
    
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Car num: {:3d}, Average Training Loss {:.5f}'.format(round, len(idxs_users), loss_avg))
            train_loss_list.append(loss_avg)
    
            # validation part
            #metric_results, iter_val_loss = test_beam_select(net_glob, dataset_val, args)
            
            # TODO:
            """val_loader = DataLoader(dataset_val, batch_size=args.local_bs, shuffle=False, drop_last=False, collate_fn=model_utils.my_collate_fn)
            round_val_loss = validate(args,
                                    args.device,
                                    val_loader,
                                    round,
                                    nn.MSELoss(),
                                    net_glob['EncoderRNN'],
                                    net_glob['DecoderRNN'],
                                    model_utils,
                                    best_loss=min_ignore_None(val_loss_list),
                                    save_address_id=args.save_address_id)
            metric_results = evaluate(args, args.device, round, data_dict, net_glob['EncoderRNN'], net_glob['DecoderRNN'], model_utils)
            val_loss_list.append(round_val_loss)
            eval_metrices_list.append(metric_results)
            print("Validation Loss:{}, Metrices: {}".format(round_val_loss ,metric_results))
        plot_loss_acc_curve(args, train_loss_list, val_loss_list, eval_metrices_list, rounds)
        save_training_log(args, train_loss_list, val_loss_list, eval_metrices_list)
    plot_loss_acc_curve(args, train_loss_list, val_loss_list, eval_metrices_list, rounds)"""

    # test part
    net_glob.eval()

    """acc_train, train_loss_list = test_beam_select(net_glob, dataset_train, args) 
    eval_metrices_list, val_loss_list = test_beam_select(net_glob, dataset_val, args) 
    acc_test, loss_test = test_beam_select(net_glob, dataset_test, args)    
    [top1_acc_train, top5_acc_train, top10_acc_train] = acc_train
    [top1_acc_val, top5_acc_val, top10_acc_val] = eval_metrices_list
    [top1_acc_test, top5_acc_test, top10_acc_test] = acc_test
    print("Training accuracy: Top-1:{:.4f}% Top-5:{:.4f}% Top-10:{:.4f}%".format(top1_acc_train * 100., top5_acc_train * 100., top10_acc_train * 100.))
    print("Validation accuracy: Top-1:{:.4f}% Top-5:{:.4f}% Top-10:{:.4f}%".format(top1_acc_val * 100., top5_acc_val * 100., top10_acc_val * 100.))
    print("Testing accuracy: Top-1:{:.4f}% Top-5:{:.4f}% Top-10:{:.4f}%".format(top1_acc_test * 100., top5_acc_test * 100., top10_acc_test * 100.))"""