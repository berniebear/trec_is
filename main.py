from __future__ import print_function, absolute_import, division, unicode_literals

import os
import random
import numpy as np

from options import get_arguments
from preprocess import Preprocess
from train import Train
from evaluate import evaluate
from utils import set_logging
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
    tweet_file_list = [os.path.join(args.data_dir, 'all-tweets.txt')]  # Tweets got by TREC jar API
    if args.use_tweets_by_API:
        tweet_file_list = [os.path.join(args.data_dir, '{}-tweets.txt'.format(part)) for part in ['train', 'test']]
    train_file = os.path.join(args.data_dir, 'TRECIS-CTIT-H-Training.json')
    test_file = os.path.join(args.data_dir, 'TRECIS-CTIT-H-Test.tweetids.tsv')
    test_label_file_list = [os.path.join(args.data_dir, 'TRECIS-2018-TestEvents-Labels', 'assr{}.test'.format(i)) for i in range(1, 7)]

    # As the original files provided by TREC is quite messy, we formalize them into train and test file
    formal_train_file = os.path.join(args.data_dir, '2018-train.txt')
    formal_test_file = os.path.join(args.data_dir, '2018-test.txt')
    utils.formalize_train_file(train_file, formal_train_file)
    utils.formalize_test_file(test_label_file_list, formal_test_file)
    if args.cross_validate:
        formal_merge_file = os.path.join(args.data_dir, '2018-all.txt')
        utils.merge_files([formal_train_file, formal_test_file], formal_merge_file)
        formal_train_file = formal_merge_file
        logger.info("Use cross-validation, the train file has been setting to {}".format(formal_train_file))

    # Files for external feature extraction (such as skip-thought and BERT)
    args.tweet_text_out_file = os.path.join(args.out_dir, 'tweets-clean-text.txt')
    args.tweet_id_out_file = os.path.join(args.out_dir, 'tweets-id.txt')
    args.skipthought_vec_file = os.path.join(args.out_dir, 'skip-thought-vec.npy')
    args.bert_vec_file = os.path.join(args.out_dir, 'bert-vec.json')

    # Step1. Preprocess and extract features for all tweets
    label2id, majority_label, short2long_label = utils.get_label2id(label_file, formal_train_file, args.cv_num)
    id2label = utils.get_id2label(label2id)
    tweetid_list, tweet_content_list = utils.get_tweetid_content(tweet_file_list)
    if not (os.path.isfile(args.tweet_text_out_file) and os.path.isfile(args.tweet_id_out_file)):
        utils.write_tweet_and_ids(tweetid_list, tweet_content_list, args.tweet_text_out_file, args.tweet_id_out_file)
    preprocess = Preprocess(args, tweetid_list, tweet_content_list, label2id)
    preprocess.extract_features()

    # Step2. Train or Cross-validation
    data_x, data_y = preprocess.extract_train_data(formal_train_file)
    train = Train(args, data_x, data_y)
    train.train()

    # Step3. Predict
    predict_file = os.path.join(args.out_dir, "predict.txt")
    test_data_x, test_tweetid_list, tweetid2idx, tweetid2incident = preprocess.extract_test_data(test_file)
    train.predict(test_data_x, test_tweetid_list, tweetid2idx, tweetid2incident,
                  id2label, short2long_label, majority_label, predict_file)
    utils.gzip_compress_file(predict_file)
    evaluate(test_label_file_list, predict_file + ".gz", label_file, args.out_dir)


if __name__ == '__main__':
    main()
