from __future__ import print_function, absolute_import, division, unicode_literals

import os
import random
import numpy as np

from options import get_arguments
from preprocess import Preprocess
from train import Train
from utils import print_to_log, set_logging
import utils


def main():
    args = get_arguments()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)  # sklearn use np to generate random value

    # set logging
    args.model_dir = os.path.join(args.out_dir, 'ckpt')
    args.log_dir = os.path.join(args.out_dir, 'log')
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.isdir(args.test_dir):
        os.mkdir(args.test_dir)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    logger = set_logging(args)

    # set test mode arguments
    if args.test:
        logger.info("Use test mode, with small dataset")
        args.data_file = args.data_file + '_test'

    print_to_log("Here is the arguments of this running:")
    print_to_log("{}".format(args))

    # Preprocess
    label_file = os.path.join(args.data_dir, 'ITR-H.types.v2.json')
    label2id = utils.get_label2id(label_file)
    tweet_file_list = [os.path.join(args.data_dir, '{}-tweets.txt'.format(it_name)) for it_name in ['train', 'test']]
    tweetid2content = utils.get_tweetid2content(tweet_file_list)
    preprocess = Preprocess(args, tweetid2content, label2id)

    train_file = os.path.join(args.data_dir, 'TRECIS-CTIT-H-Training.json')
    test_file_list = [os.path.join(args.data_dir, 'assr{}.test'.format(i)) for i in range(7)]
    preprocess.extract_train_data(train_file)
    data_x, data_y = preprocess.content_to_feature()

    # Train
    train = Train(args, data_x, data_y)


if __name__ == '__main__':
    main()
