from __future__ import print_function, absolute_import, division, unicode_literals

import os
import random
import numpy as np

from options import get_arguments
from preprocess import Preprocess
from train import Train
from evaluate import evaluate
from utils import set_logging, prepare_folders
import utils


def main():
    args = get_arguments()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)  # sklearn use np to generate random value

    # Create folders and set logging format
    args.model_dir = os.path.join(args.out_dir, 'ckpt')
    args.log_dir = os.path.join(args.out_dir, 'log')
    args.ensemble_dir = os.path.join(args.out_dir, 'ensemble')
    prepare_folders(args)
    logger = set_logging(args)
    logger.info("Here is the arguments of this running:")
    logger.info("{}".format(args))
    utils.check_args_conflict(args)

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

    # Step1. Preprocess and extract features for all tweets
    label2id, majority_label, short2long_label = utils.get_label2id(label_file, formal_train_file, args.cv_num)
    id2label = utils.get_id2label(label2id)
    tweetid_list, tweet_content_list = utils.get_tweetid_content(tweet_file_list)
    if not (os.path.isfile(args.tweet_text_out_file) and os.path.isfile(args.tweet_id_out_file)):
        utils.write_tweet_and_ids(tweetid_list, tweet_content_list, args.tweet_text_out_file, args.tweet_id_out_file)
    preprocess = Preprocess(args, tweetid_list, tweet_content_list, label2id)
    preprocess.extract_features()

    # Step2. Train and Cross-validation
    data_x, data_y = preprocess.extract_train_data(formal_train_file)
    if args.event_wise:
        metrics_collect = []
        metric_names = None
        for event_type in utils.idx2event_type:
            it_data_x, it_data_y = data_x[event_type], data_y[event_type]
            train = Train(args, it_data_x, it_data_y, id2label, preprocess.feature_len, event_type)
            train.shuffle_data()
            metrics = train.train()
            metrics_collect.append((metrics, it_data_x.shape[0]))
            if metric_names is None:
                metric_names = train.metric_names
        utils.get_final_metrics(metrics_collect, metric_names)
    else:
        train = Train(args, data_x, data_y, id2label, preprocess.feature_len)
        train.shuffle_data()
        train.train()

    if args.predict_mode:
        # Step3. Get the 2019 test data, and retrain the model on all training data, then predict on the 2019-test
        # Todo: Generate the formalized file after 2019-test data released
        # Todo: After 2019-test data released, add it to tweet_file_list to make sure those tweets have features
        # Todo: Convert label to the new long-label for 2019 setting
        formal_2019_test_file = os.path.join(args.data_dir, '2019-test.txt')
        if args.event_wise:
            test_x, event2idx_list, line_num = preprocess.extract_formalized_test_data(formal_2019_test_file)
            test_predict_collect = [0] * line_num
            for event_type in utils.idx2event_type:
                it_data_x, it_data_y, it_test_x = data_x[event_type], data_y[event_type], test_x[event_type]
                train = Train(args, it_data_x, it_data_y, id2label, preprocess.feature_len, event_type)
                train.train_on_all()
                predict_score = train.predict_on_test(it_test_x)
                for i, idx in event2idx_list[event_type]:
                    test_predict_collect[idx] = predict_score[i]
            test_predict_file = os.path.join(args.ensemble_dir, 'test_predict_{0}.txt'.format(args.model))
            with open(test_predict_file, 'w', encoding="utf8") as f:
                for row in test_predict_collect:
                    f.write('{}\n'.format(' '.join(list(map(str, row)))))
            print("The test predict has been written to {0}".format(test_predict_file))
        else:
            test_x = preprocess.extract_formalized_test_data(formal_2019_test_file)
            train = Train(args, data_x, data_y, id2label, preprocess.feature_len)
            train.train_on_all()
            train.predict_on_test(test_x)

    # The old predict script for evaluation on 2018-test data
    # predict_file = os.path.join(args.out_dir, "predict.txt")
    # test_data_x, test_tweetid_list, tweetid2idx, tweetid2incident = preprocess.extract_test_data(test_file)
    # train.predict(test_data_x, test_tweetid_list, tweetid2idx, tweetid2incident,
    #               id2label, short2long_label, majority_label, predict_file)
    # utils.gzip_compress_file(predict_file)
    # evaluate(test_label_file_list, predict_file + ".gz", label_file, args.out_dir)


if __name__ == '__main__':
    main()
