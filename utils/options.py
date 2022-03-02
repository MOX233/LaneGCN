#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # cash
    parser.add_argument("--save_address_id", type=str, default="default") # identify saving path. Need to change in the future


    # federated arguments
    parser.add_argument('--num_items', type=int, default=8096,
                        help="number of data from every user's local dataset. type: int or list")
    parser.add_argument('--local_train_speed', type=float, default=10,
                        help="the calculation speed of local iteration. local_train_speed * local_train_time = local_iter")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="SGD momentum (default: 0.5)")

    # model arguments
    parser.add_argument("--model",
                        type=str,
                        default="lanegcn",
                        help="DL model")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    
    # training arguments only used in lstm_run.py
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=128,
                        help="Training batch size")
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=128,
                        help="Test batch size")
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=128,
                        help="Val batch size")
    parser.add_argument("--end_epoch",
                        type=int,
                        default=360,
                        help="Last epoch")
    parser.add_argument("--resume", default="", type=str, metavar="RESUME", help="checkpoint path")
    parser.add_argument("--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path")

    # evaluation arguments
    parser.add_argument("--metrics",
                        action="store_true",
                        help="If true, compute metrics")
    parser.add_argument("--gt", default="", type=str, help="path to gt file")
    parser.add_argument("--miss_threshold",
                        default=2.0,
                        type=float,
                        help="Threshold for miss rate")
    parser.add_argument("--max_n_guesses",
                        default=0,
                        type=int,
                        help="Max number of guesses")
    parser.add_argument("--prune_n_guesses",
                        default=0,
                        type=int,
                        help="Pruned number of guesses of non-map baseline using map",)
    parser.add_argument("--n_guesses_cl",
                        default=0,
                        type=int,
                        help="Number of guesses along each centerline",)
    parser.add_argument("--n_cl",
                        default=0,
                        type=int,
                        help="Number of centerlines to consider")
    parser.add_argument("--viz",
                        action="store_true",
                        help="If true, visualize predictions")
    parser.add_argument("--viz_seq_id",
                        default="",
                        type=str,
                        help="Sequence ids for the trajectories to be visualized",)
    parser.add_argument("--max_neighbors_cl",
                        default=3,
                        type=int,
                        help="Number of neighbors obtained for each centerline by the baseline",)

    # other arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str,
                        default='Argoverse', help="name of dataset")
    parser.add_argument('--non_iid', action='store_true',
                        default=False,  help='whether i.i.d. or not')
    parser.add_argument('--verbose', action='store_true',
                        default=False,  help='verbose print')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed which make tests reproducible (default: 1)')
    parser.add_argument('--plot_save_path', type=str, default="default",
                        help="The save path for the plots of loss and other metrices.")
    parser.add_argument('--log_save_path', type=str, default="default",
                        help="The save path for the training log of loss and other metrices.")
    parser.add_argument("--traj_save_path",
                        required=False,
                        type=str,
                        help="path to the pickle file where forecasted trajectories will be saved.",)

    # dataset argumrnt by Argoverse Motion Forecasting
    parser.add_argument("--train_features",
                        default="",
                        type=str,
                        help="path to the file which has train features.",
                        )
    parser.add_argument("--val_features",
                        default="",
                        type=str,
                        help="path to the file which has val features.",
                        )
    parser.add_argument("--test_features",
                        default="",
                        type=str,
                        help="path to the file which has test features.",)
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")

    # SUMO arguments
    parser.add_argument("--sumo_data_dir", type=str, 
                         default="./sumo_data", help="the directory where saves the necessary config files for SUMO running")
    parser.add_argument("--no_sumo_run", action="store_true",
                        default=False, help="run sumo simulation to generate tripinfo.xml")
    parser.add_argument("--nogui", action="store_true",
                        default=True, help="run the commandline version of sumo")
    parser.add_argument("--trajectoryInfo_path", type=str,
                        default='./sumo_result/trajectory.csv', help="the file path where stores the trajectory infomation of cars")
    parser.add_argument("--step_length", type=float,
                        default=0.1, help="sumo sampling interval")
    parser.add_argument("--num_steps", type=int,
                        default=10000, help="number of time steps, which means how many seconds the car flow takes")
    parser.add_argument("--round_duration", type=float,
                        default=100, help="duration time of each round")
    parser.add_argument("--beta_download", type=float,
                        default=1, help="param of shift exponential distribution function for download delay")
    parser.add_argument("--beta_upload", type=float,
                        default=1, help="param of shift exponential distribution function for upload delay")
    parser.add_argument("--mu_download", type=float,
                        default=1, help="param of shift exponential distribution function for download delay")
    parser.add_argument("--mu_upload", type=float,
                        default=1, help="param of shift exponential distribution function for upload delay")
    parser.add_argument("--local_train_time", type=float,
                        default=5, help="local training time for each vehicle")
    parser.add_argument("--Lambda", type=float,
                        default=0.1, help="arrival rate of car flow")
    parser.add_argument("--accel", type=float,
                        default=10, help="accelerate of car flow")
    parser.add_argument("--decel", type=float,
                        default=20, help="decelerate of car flow")
    parser.add_argument("--sigma", type=float,
                        default=0, help="imperfection of drivers, which takes value on [0,1], with 0 meaning perfection and 1 meaning imperfection")
    parser.add_argument("--carLength", type=float,
                        default=5, help="length of cars")
    parser.add_argument("--minGap", type=float,
                        default=2.5, help="minimum interval between adjacent cars")
    parser.add_argument("--maxSpeed", type=float,
                        default=20, help="maxSpeed for cars")
    parser.add_argument("--speedFactoer_mean", type=float,
                        default=1, help="")
    parser.add_argument("--speedFactoer_dev", type=float,
                        default=0.1, help="")
    parser.add_argument("--speedFactoer_min", type=float,
                        default=0.5, help="")
    parser.add_argument("--speedFactoer_max", type=float,
                        default=1.5, help="")
    args = parser.parse_args()
    args.local_iter = args.local_train_time * args.local_train_speed
    return args
