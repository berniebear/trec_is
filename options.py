import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # test mode
    parser.add_argument("--test", action='store_true',
                        help="Use small data set to run")
    # reproducibility
    parser.add_argument("--random_seed", type=int, default=9,
                        help="Random seed (>0, set a specific seed).")
    # Data path
    parser.add_argument("--out_dir", type=str, default='out',
                        help="Directory contains the output things, including the log and ckpt")
    parser.add_argument("--test_dir", type=str, default='test',
                        help="Directory contains the small dataset for testing")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="Directory contains the data")
    parser.add_argument("--log_name", type=str, default='defaultLog',
                        help="Name of logfile, could use Date such as Apr25")
    # Model
    parser.add_argument("--model", type=str, default='bernoulli_nb',
                        help="bernoulli_nb | rf | sgd")
    parser.add_argument("--no_class_weight", action='store_true',
                        help="Not use different weights for each class")
    parser.add_argument("--cv_num", type=int, default=5,
                        help="The n-fold cross-validation")
    return parser.parse_args()
