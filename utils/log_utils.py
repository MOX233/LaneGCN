#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import os
import pickle as pkl

def save_training_log(args, train_loss_list, val_loss_list, eval_metrices_list):

    log_dict = {}
    log_dict['loss_train'] = train_loss_list
    log_dict['loss_val'] = val_loss_list
    log_dict['metrices_eval'] = eval_metrices_list
    log_dict.update(vars(args))
    savePath = "./save"
    if args.log_save_path != "default":
            savePath = args.log_save_path
    os.makedirs(savePath, exist_ok=True)
    savePath = os.path.join(savePath,'RoundDuration{}_LocalTrainDelay_mu{}_beta{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}_noniid{}.pkl'.format(args.round_duration, args.mu_local_train, args.beta_local_train, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed, args.non_iid))
    with open(savePath,'wb') as f:
        pkl.dump(log_dict, f)

def load_training_log(savePath):
    with open(savePath,'rb') as f:
        log_dict = pkl.load(f)
    return log_dict