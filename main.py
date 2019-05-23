from __future__ import print_function, absolute_import, division, unicode_literals

import os
import random
import numpy as np

from options import get_arguments
from preprocess import Preprocess
from train import Train
from postprocess import PostProcess
from utils import set_logging, prepare_folders
import utils


def main():
    args = get_arguments()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)  # sklearn use np to generate random value

    # Create folders and set logging format
    args.model_dir = os.path.join(args.out_dir, 'ckpt-{}'.format(args.class_weight_scheme))
    args.log_dir = os.path.join(args.out_dir, 'log')
    args.ensemble_dir = os.path.join(args.out_dir, 'ensemble-{}'.format(args.class_weight_scheme))
    prepare_folders(args)
    logger = set_logging(args)
    logger.info("Here is the arguments of this running:")
    logger.info("{}".format(args))
    utils.check_args_conflict(args)

    # Set the file contains data for training and test
    label_file = os.path.join(args.data_dir, 'ITR-H.types.v2.json')
    tweet_file_list = [os.path.join(args.data_dir, 'all-tweets.txt')]
    tweet_file_list_2019 = [os.path.join(args.data_dir, 'all-tweets-2019.txt')]
    train_file = os.path.join(args.data_dir, 'TRECIS-CTIT-H-Training.json')
    test_label_file_list = [os.path.join(args.data_dir, 'TRECIS-2018-TestEvents-Labels', 'assr{}.test'.format(i)) for i in range(1, 7)]
    formal_train_file = os.path.join(args.data_dir, '2018-train.txt')
    formal_test_file = os.path.join(args.data_dir, '2018-test.txt')
    formal_2019_test_file = os.path.join(args.data_dir, '2019-test.txt')
    tweet_text_out_file = os.path.join(args.out_dir, 'tweets-clean-text.txt')
    tweet_id_out_file = os.path.join(args.out_dir, 'tweets-id.txt')
    tweet_text_out_file_2019 = os.path.join(args.out_dir, 'tweets-clean-text-2019.txt')
    tweet_id_out_file_2019 = os.path.join(args.out_dir, 'tweets-id-2019.txt')
    raw_tweets_json_folder = 'download_tweets'

    # As the original files provided by TREC is quite messy, we formalize them into train and test file
    utils.formalize_train_file(train_file, formal_train_file)
    utils.formalize_test_file(test_label_file_list, formal_test_file)
    utils.formalize_2019_test_file(raw_tweets_json_folder, formal_2019_test_file)
    if args.cross_validate:
        formal_merge_file = os.path.join(args.data_dir, '2018-all.txt')
        utils.merge_files([formal_train_file, formal_test_file], formal_merge_file)
        formal_train_file = formal_merge_file
        logger.info("Use cross-validation, the train file has been setting to {}".format(formal_train_file))

    # Step0. Extract some info, which is necessary to generate submission files
    # (must run cross-validate and predict_mode before run the submission mode)
    label2id, majority_label, short2long_label = utils.get_label2id(label_file, formal_train_file, args.cv_num)
    id2label = utils.get_id2label(label2id)
    class_weight = utils.get_class_weight(label2id, id2label, formal_train_file)
    if args.get_submission:
        postpro = PostProcess(args, label2id, id2label, class_weight, majority_label, short2long_label,
                              formal_train_file, formal_2019_test_file, raw_tweets_json_folder)
        if args.pick_criteria == 'threshold':
            threshold = postpro.find_best_threshold() if args.pick_threshold is None else args.pick_threshold
            postpro.pick_by_threshold(threshold)
        elif args.pick_criteria == 'top':
            postpro.pick_top_k(args.pick_k)
        else:
            postpro.pick_by_autothre()
        quit()

    # Step1. Preprocess and extract features for all tweets
    tweetid_list, tweet_content_list = utils.get_tweetid_content(tweet_file_list)
    utils.write_tweet_and_ids(tweetid_list, tweet_content_list, tweet_text_out_file, tweet_id_out_file)
    tweetid_list_2019, tweet_content_list_2019 = utils.get_tweetid_content(tweet_file_list_2019)
    utils.write_tweet_and_ids(tweetid_list_2019, tweet_content_list_2019, tweet_text_out_file_2019,
                              tweet_id_out_file_2019)
    preprocess = Preprocess(args, tweetid_list, tweet_content_list, label2id, tweet_id_out_file)
    preprocess.extract_features()
    preprocess_2019 = Preprocess(args, tweetid_list_2019, tweet_content_list_2019, label2id,
                                 tweet_id_out_file_2019, test=True)
    preprocess_2019.extract_features()

    # Step2. Train and Cross-validation
    if args.event_wise:
        data_x, data_y, event2idx_list, line_num = preprocess.extract_train_data(formal_train_file)
        data_predict_collect = np.zeros([line_num, len(label2id)])
        metrics_collect = []
        metric_names = None
        for event_type in utils.idx2event_type:
            it_data_x, it_data_y = data_x[event_type], data_y[event_type]
            train = Train(args, it_data_x, it_data_y, id2label, preprocess.feature_len, class_weight, event_type)
            metrics, predict_score = train.train()
            for i, idx in enumerate(event2idx_list[event_type]):
                data_predict_collect[idx] = predict_score[i]
            metrics_collect.append((metrics, it_data_x.shape[0]))
            if metric_names is None:
                metric_names = train.metric_names
        utils.get_final_metrics(metrics_collect, metric_names)
    else:
        data_x, data_y = preprocess.extract_train_data(formal_train_file)
        train = Train(args, data_x, data_y, id2label, preprocess.feature_len, class_weight)
        _, data_predict_collect = train.train()
    if args.predict_mode:
        utils.write_predict_and_label(args, formal_train_file, label2id, data_predict_collect)

    if args.predict_mode:
        # Step3. Get the 2019 test data, and retrain the model on all training data, then predict on the 2019-test
        if args.event_wise:
            test_x, event2idx_list, line_num = preprocess_2019.extract_formalized_test_data(formal_2019_test_file)
            test_predict_collect = np.zeros([line_num, len(label2id)])
            for event_type in utils.idx2event_type:
                it_data_x, it_data_y, it_test_x = data_x[event_type], data_y[event_type], test_x[event_type]
                if len(it_test_x) == 0:
                    print("[WARNING] There are no event belongs to {} for the test data".format(event_type))
                    continue
                train = Train(args, it_data_x, it_data_y, id2label, preprocess_2019.feature_len, class_weight, event_type)
                train.train_on_all()
                predict_score = train.predict_on_test(it_test_x)
                for i, idx in enumerate(event2idx_list[event_type]):
                    test_predict_collect[idx] = predict_score[i]
        else:
            test_x = preprocess_2019.extract_formalized_test_data(formal_2019_test_file)
            train = Train(args, data_x, data_y, id2label, preprocess_2019.feature_len, class_weight)
            train.train_on_all()
            test_predict_collect = train.predict_on_test(test_x)
        utils.write_predict_res_to_file(args, test_predict_collect)

    if args.ensemble is not None:
        # Step4. Do the ensemble of different model
        if args.event_wise:
            raise NotImplementedError("We don't want to ensemble for event-wise models")
        else:
            out_file = os.path.join(args.out_dir, 'ensemble_out.txt')
            dev_label_file = os.path.join(args.ensemble_dir, 'dev_label.txt')
            dev_predict_file_list = utils.get_predict_file_list(args.ensemble_dir, 'dev_predict_')
            test_predict_file_list = utils.get_predict_file_list(args.ensemble_dir, 'test_predict_')
            train_x = utils.get_ensemble_feature(dev_predict_file_list)
            train_y = utils.get_ensemble_label(dev_label_file)
            print("The shape of ensemble train_x is {0}".format(train_x.shape))
            utils.ensemble_cross_validate(train_x, train_y, id2label, train.mlb, args.ensemble)
            test_x = utils.get_ensemble_feature(test_predict_file_list)
            predict = utils.ensemble_train_and_predict(train_x, train.mlb.transform(train_y), test_x,
                                                       id2label, args.ensemble)
            predict = [id2label[x] for x in predict]
            with open(out_file, 'w', encoding='utf8') as f:
                for it_predict in predict:
                    f.write("{}\n".format(it_predict))
            print("The ensemble result has been written to {}".format(out_file))


if __name__ == '__main__':
    main()
