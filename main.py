from __future__ import print_function, absolute_import, division, unicode_literals

import os
import random
import numpy as np

from options import get_arguments
from preprocess import Preprocess
from train import Train, TrainRegression
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
    if args.class_weight_scheme == 'customize':
        args.model_dir = os.path.join(args.model_dir, 'weight{}'.format(args.additional_weight))
        args.ensemble_dir = os.path.join(args.ensemble_dir, 'weight{}'.format(args.additional_weight))
    prepare_folders(args)
    logger = set_logging(args)
    logger.info("Here is the arguments of this running:")
    logger.info("{}".format(args))
    utils.check_args_conflict(args)

    # Set files which contain data for training and test. If use "trecis2019-A", it means we want to tune parameters.
    args.data_prefix = "trecis2019-B"
    # Note that for 2019-B submission, all '2019' means '2019-B' and '2018' means '2018 + 2019-A'
    label_file = os.path.join(args.data_dir, 'ITR-H.types.v{}.json'.format(
        4 if args.data_prefix == "trecis2019-B" else 3))
    tweet_file_list = [os.path.join(args.data_dir, 'all-tweets.txt')]
    tweet_file_list_2019 = [os.path.join(args.data_dir, 'all-tweets-2019.txt')]
    train_file_list = [os.path.join(args.data_dir, 'TRECIS-CTIT-H-Training.json')]
    train_file_list += [os.path.join(args.data_dir, 'TRECIS-2018-TestEvents-Labels',
                                     'assr{}.test'.format(i)) for i in range(1, 7)]
    if args.data_prefix == "trecis2019-B":
        train_file_list += [os.path.join(args.data_dir, '2019ALabels', '2019A-assr{}.json'.format(i)) for i in range(1, 6)]
        train_file_list += [os.path.join(args.data_dir, '2019ALabels', '2019-assr2.json')]
    test_raw_tweets_json_folder = 'download_tweets'
    # Some output files which has been formalized for further usages.
    formal_train_file = os.path.join(args.data_dir, 'train.txt')
    formal_test_file = os.path.join(args.data_dir, 'test.txt')
    tweet_text_out_file = os.path.join(args.out_dir, 'tweets-clean-text.txt')
    tweet_id_out_file = os.path.join(args.out_dir, 'tweets-id.txt')
    tweet_text_out_file_2019 = os.path.join(args.out_dir, 'tweets-clean-text-2019.txt')
    tweet_id_out_file_2019 = os.path.join(args.out_dir, 'tweets-id-2019.txt')
    predict_priority_score_out_file = os.path.join(args.out_dir, 'predict_priority_score.txt')

    # Set files for submission.
    args.model_name = '{0}{1}'.format(args.model, '-event' if args.event_wise else '')
    args.dev_label_file = os.path.join(args.ensemble_dir, 'dev_label.txt')
    args.dev_predict_file = os.path.join(args.ensemble_dir, 'dev_predict_{}.txt'.format(args.model_name))
    args.test_predict_file = os.path.join(args.ensemble_dir, 'test_predict_{}.txt'.format(args.model_name))
    args.submission_folder = utils.prepare_submission_folder(args)
    args.submission_file = os.path.join(args.submission_folder, 'submission_{}'.format(args.model_name))

    # As the original files provided by TREC is quite messy, we formalize them into train and test file.
    utils.formalize_files(train_file_list, formal_train_file)
    utils.formalize_test_file(test_raw_tweets_json_folder, formal_test_file, prefix=args.data_prefix)
    logger.info("The training data file is {0} and testing data file is {1}".format(
        formal_train_file, formal_test_file))

    # Step0. Extract some info which can be used later (also useful for generating submission files).
    label2id, majority_label, short2long_label = utils.get_label2id(label_file, formal_train_file, args.cv_num)
    id2label = utils.get_id2label(label2id)
    class_weight = utils.get_class_weight(args, label2id, id2label, formal_train_file)

    # When get submission, there is no need to run all following steps, but only read the `test_predict_file` and
    # pick some classes as final output according to policy (such as top-2 or auto-threshold).
    # You MUST run `--predict_mode` in advance to get the `test_predict_file` prepared.
    if args.get_submission:
        postpro = PostProcess(args, label2id, id2label, class_weight, majority_label, short2long_label,
                              formal_train_file, formal_test_file, test_raw_tweets_json_folder,
                              predict_priority_score_out_file)
        postpro.pick_labels_and_write_final_result()
        quit()

    # Step1. Preprocess and extract features for all tweets
    tweetid_list, tweet_content_list = utils.get_tweetid_content(tweet_file_list)
    utils.write_tweet_and_ids(tweetid_list, tweet_content_list, tweet_text_out_file, tweet_id_out_file)
    tweetid_list_2019, tweet_content_list_2019 = utils.get_tweetid_content(tweet_file_list_2019)
    utils.write_tweet_and_ids(tweetid_list_2019, tweet_content_list_2019, tweet_text_out_file_2019,
                              tweet_id_out_file_2019)
    # Note that before `extract_features()`, we should manually run the `extract_features.sh` in `feature_tools`.
    # quit()  # The `extract_features.sh` only need to be run once for the same dataset.
    preprocess = Preprocess(args, tweetid_list, tweet_content_list, label2id, tweet_id_out_file)
    preprocess.extract_features()
    preprocess_2019 = Preprocess(args, tweetid_list_2019, tweet_content_list_2019, label2id,
                                 tweet_id_out_file_2019, test=True)
    preprocess_2019.extract_features()

    if args.train_regression:
        data_x, data_score = preprocess.extract_train_data(formal_train_file, get_score=True)
        train_regression = TrainRegression(args, data_x, data_score)
        if args.cross_validate:
            train_regression.cross_validate()
            quit()

    if args.cross_validate:
        # Step2. Train and Cross-validation (for tuning hyper-parameters).
        # If we want to do ensemble in the future, we need the prediction on dev data by setting `--cross_validate`.
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
            data_x, data_y, _, _ = preprocess.extract_train_data(formal_train_file)
            test_x, event2idx_list, line_num = preprocess_2019.extract_formalized_test_data(formal_test_file)
            test_predict_collect = np.zeros([line_num, len(label2id)])
            for event_type in utils.idx2event_type:
                it_data_x, it_data_y, it_test_x = data_x[event_type], data_y[event_type], test_x[event_type]
                if len(it_test_x) == 0:
                    print("[WARNING] There are no event belongs to {} for the test data".format(event_type))
                    continue
                train = Train(args, it_data_x, it_data_y, id2label,
                              preprocess_2019.feature_len, class_weight, event_type)
                train.train_on_all()
                predict_score = train.predict_on_test(it_test_x)
                for i, idx in enumerate(event2idx_list[event_type]):
                    test_predict_collect[idx] = predict_score[i]
        else:
            data_x, data_y = preprocess.extract_train_data(formal_train_file)
            test_x = preprocess_2019.extract_formalized_test_data(formal_test_file)
            train = Train(args, data_x, data_y, id2label, preprocess_2019.feature_len, class_weight)
            train.train_on_all()
            test_predict_collect = train.predict_on_test(test_x)
        utils.write_predict_res_to_file(args, test_predict_collect)

        if args.train_regression:
            test_x = preprocess_2019.extract_formalized_test_data(formal_test_file)
            if args.event_wise:
                # For event_wise setting, there will be many additional things extracted, what we need is only test_x.
                test_x = test_x[0]
            train_regression.train()
            predict_priority_score = train_regression.predict_on_test(test_x)
            utils.write_predict_score_to_file(predict_priority_score, predict_priority_score_out_file)

    if args.ensemble is not None:
        # TODO(junpeiz): Average the priority score for ensemble.
        # Step4 (optional). Do the ensemble of different model
        if args.event_wise:
            raise NotImplementedError("We don't want to ensemble for event-wise models")
        else:
            out_file = os.path.join(args.out_dir, 'ensemble_out.txt')
            # Note the file list contains predictions from all models with and without the '-event' suffix.
            # So, we need to train both event-wise and not event-wise models or just delete those files in the folder.
            dev_predict_file_list = utils.get_predict_file_list(args.ensemble_dir, 'dev_predict_')
            test_predict_file_list = utils.get_predict_file_list(args.ensemble_dir, 'test_predict_')
            train_x = utils.get_ensemble_feature(dev_predict_file_list)
            train_y = utils.get_ensemble_label(args.dev_label_file)
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
