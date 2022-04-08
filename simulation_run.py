
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

from utils.sumo_utils import read_tripInfo, sumo_run, sumo_run_with_trajectoryInfo
from utils.interface_for_FL import generate_FLtable_from_tripInfo
from utils.options import args_parser
from utils.FL_training import FL_training

# this is the main entry point of this script
if __name__ == "__main__":
    args = args_parser()
    args.MU_local_train = args.local_iter * args.mu_local_train
    args.BETA_local_train = args.local_iter * args.beta_local_train

    if args.no_sumo_run == False:
        sumo_run(args, save_dir=args.sumo_data_dir)
    car_tripinfo = read_tripInfo(tripInfo_path=os.path.join(args.sumo_data_dir,'tripinfo.xml'))
    FL_table = generate_FLtable_from_tripInfo(args)

    file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(file_path)

    if args.save_address_id == "default":
        args.save_address_id = 'RoundDuration{}_LocalTrainDelay_mu{}_beta{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}_noniid{}'.format(args.round_duration, args.mu_local_train, args.beta_local_train, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed, int(args.non_iid))
    
    #args.local_iter = int(args.local_train_speed * args.local_train_time)

    FL_training(args,FL_table,car_tripinfo)