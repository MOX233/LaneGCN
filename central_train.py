
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

from utils.sumo_utils import read_tripInfo, sumo_run, sumo_run_with_trajectoryInfo
from utils.interface_for_FL import generate_FLtable_from_tripInfo
from utils.options import args_parser
from utils.CL_training import CL_training

# this is the main entry point of this script
if __name__ == "__main__":
    args = args_parser()
    args.MU_local_train = args.local_iter * args.mu_local_train
    args.BETA_local_train = args.local_iter * args.beta_local_train

    city = 'PIT'
    file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(file_path)


    if args.save_address_id == "default":
        args.save_address_id = 'CL_training_' + city
    
    CL_training(args,city=city)