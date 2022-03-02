#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
from genericpath import exists
import os
import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa


def min_ignore_None(data_list):
    data_list_ignore_None = []
    for data in data_list:
        if data != None:
            data_list_ignore_None.append(data)
    if len(data_list_ignore_None) == 0:
        return None
    else:
        return min(data_list_ignore_None)

def max_ignore_None(data_list):
    data_list_ignore_None = []
    for data in data_list:
        if data != None:
            data_list_ignore_None.append(data)
    if len(data_list_ignore_None) == 0:
        return None
    else:
        return max(data_list_ignore_None)


def plot_loss_acc_curve(args, loss_train, loss_val, metrices_eval, rounds):
    plt.figure(figsize=(15, 15), dpi=100)
    # loss
    plt.subplot(2, 2, 1)
    plt.plot(range(len(loss_train)), loss_train,
             color='r', label='Training Loss')
    plt.plot(range(len(loss_val)), loss_val,
             color='b', label='Validation Loss')
    plt.xlabel('round')
    plt.ylabel('train_loss')
    plt.legend()
    # accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(len(metrices_eval)), [i["minADE"]
             for i in metrices_eval], color='r', label='minADE')
    plt.plot(range(len(metrices_eval)), [i["minFDE"]
             for i in metrices_eval], color='g', label='minFDE')
    plt.xlabel('round')
    plt.ylabel('minADE/minFDE')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(range(len(metrices_eval)), [i["MR"]
             for i in metrices_eval], color='b', label='MR')
    plt.plot(range(len(metrices_eval)), [i["DAC"]
             for i in metrices_eval], color='r', label='DAC')
    plt.xlabel('round')
    plt.ylabel('MR/DAC')
    plt.legend()
    # params description
    plt.subplot(2, 2, 3)
    plt.axis([0, 10, 0, 10])
    plt.axis('off')
    fontsize = 12
    plt.text(0, 9, 'Round Num: {}'.format(rounds), fontsize=fontsize)
    plt.text(0, 8, 'Round Duration: {}'.format(
        args.round_duration), fontsize=fontsize)
    plt.text(0, 7, 'Local Train Delay: {}'.format(
        args.local_train_time), fontsize=fontsize)
    plt.text(0, 6, 'Local Iter Num: {}'.format(
        args.local_iter), fontsize=fontsize)
    plt.text(0, 5, 'Local Batch Size: {}'.format(
        args.local_bs), fontsize=fontsize)
    plt.text(0, 4, 'Learning Rate: {}'.format(args.lr), fontsize=fontsize)
    plt.text(0, 3, 'non-i.i.d.: {}'.format(args.non_iid), fontsize=fontsize)
    plt.text(0, 2, 'min_train_loss {}'.format(
        min_ignore_None(loss_train)), fontsize=fontsize)
    plt.text(0, 1, 'min_val_loss {}'.format(min_ignore_None(loss_val)), fontsize=fontsize)
    plt.text(0, 0, 'min_minADE {}'.format(
        min_ignore_None([i["minADE"] for i in metrices_eval])), fontsize=fontsize)

    plt.text(6, 9, 'Lambda: {}'.format(args.Lambda), fontsize=fontsize)
    plt.text(6, 8, 'maxSpeed: {}'.format(args.maxSpeed), fontsize=fontsize)
    plt.text(6, 7, 'beta_download: {}'.format(
        args.beta_download), fontsize=fontsize)
    plt.text(6, 6, 'mu_download: {}'.format(
        args.mu_download), fontsize=fontsize)
    plt.text(6, 5, 'beta_upload: {}'.format(
        args.beta_upload), fontsize=fontsize)
    plt.text(6, 4, 'mu_upload: {}'.format(args.mu_upload), fontsize=fontsize)
    """if args.split_dict == 0:
        plt.text(6, 3, 'split_dict: {}'.format(
            "LOS_split_dict"), fontsize=fontsize)
    elif args.split_dict == 1:
        plt.text(6, 3, 'split_dict: {}'.format(
            "coord_split_dict"), fontsize=fontsize)
    else:
        plt.text(6, 3, 'split_dict: {}'.format(
            "unrecognized split_dict"), fontsize=fontsize)"""

    savePath = "./save"
    if args.plot_save_path != "default":
        savePath = args.plot_save_path
    os.makedirs(savePath, exist_ok=True)
    savePath = os.path.join(savePath, 'RoundDuration{}_LocalTrainDelay{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}_noniid{}.png'.format(
        args.round_duration, args.local_train_time, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed, args.non_iid))
    plt.savefig(savePath)
    plt.close()
