from __future__ import print_function, absolute_import, division, unicode_literals

import os
import random
import numpy as np

from options import get_arguments
from preprocess import Preprocess
from train import Train
from evaluate import evaluate
from utils import print_to_log, set_logging
import utils

from nltk.tokenize import TweetTokenizer
local_tokenizer = TweetTokenizer()


def tokenizer_wrapper(text):
    return local_tokenizer.tokenize(text)


def main():
    args = get_arguments()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)  # sklearn use np to generate random value

    # Set logging format and logging file
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

    logger.info("Here is the arguments of this running:")
    logger.info("{}".format(args))

    # Set the file contains data for training and test
    label_file = os.path.join(args.data_dir, 'ITR-H.types.v2.json')
    # tweet_file_list = [os.path.join(args.data_dir, 'tweets-content-merged.txt')]  # Tweets got by TREC jar API
    tweet_file_list = [os.path.join(args.data_dir, '{}-tweets.txt'.format(part)) for part in ['train', 'test']]
    train_file = os.path.join(args.data_dir, 'TRECIS-CTIT-H-Training.json')
    test_file = os.path.join(args.data_dir, 'TRECIS-CTIT-H-Test.tweetids.tsv')
    test_label_file_list = [os.path.join(args.data_dir, 'TRECIS-2018-TestEvents-Labels', 'assr{}.test'.format(i)) for i in range(1, 7)]
    predict_file = os.path.join(args.out_dir, "predict.txt")

    # Step1. Preprocess
    label2id, majority_label, short2long_label = utils.get_label2id(label_file, train_file, args.cv_num)
    id2label = utils.get_id2label(label2id)
    tweetid2content = utils.get_tweetid2content(tweet_file_list)
    preprocess = Preprocess(args, tweetid2content, label2id)

    preprocess.extract_train_data(train_file)
    data_x, data_y = preprocess.content_to_feature()

    # Step2. Train
    train = Train(args, data_x, data_y)
    train.train()

    # Step3. Predict
    test_data_x, test_tweetid_list, tweetid2idx, tweetid2incident = preprocess.extract_test_data(test_file)
    train.predict(test_data_x, test_tweetid_list, tweetid2idx, tweetid2incident,
                  id2label, short2long_label, majority_label, predict_file)
    utils.gzip_compress_file(predict_file)
    evaluate(test_label_file_list, predict_file + ".gz", label_file, args.out_dir)


if __name__ == '__main__':
    main()
