import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # sanity check
    parser.add_argument("--sanity_check", action='store_true',
                        help="Use small data set to run for sanity check")
    # reproducibility
    parser.add_argument("--random_seed", type=int, default=9,
                        help="Random seed (>0, set a specific seed).")
    # Mode
    parser.add_argument("--cross_validate", action='store_true',
                        help="Use cross-validation on the whole data (including 2018 train and test)")
    parser.add_argument("--train_on_small", action='store_true',
                        help="Use 18-train to train, and test on cross-validation test, to see if additional data helps")
    parser.add_argument("--late_fusion", action='store_true',
                        help="Don't concat features early, train models on each kind of feature, and then merge them")
    parser.add_argument("--event_wise", action='store_true',
                        help="Use event-wise classification. There are six types of events specified by the TREC-IS")
    parser.add_argument("--search_best_parameters", action='store_true',
                        help="Use random search to search for best parameters for each model")
    parser.add_argument("--random_search_n_iter", type=int, default=100,
                        help="Number of iterations to do random search (each iteration train a set of parameters)")
    # Data path
    parser.add_argument("--out_dir", type=str, default='out',
                        help="Directory contains the output things, including the log and ckpt")
    parser.add_argument("--test_dir", type=str, default='test',
                        help="Directory contains the small dataset for testing")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="Directory contains the data")
    parser.add_argument("--log_name", type=str, default='defaultLog',
                        help="Name of logfile, could use Date such as Apr25")
    parser.add_argument("--use_tweets_by_API", action='store_true',
                        help="Use the tweets file gotten by Twitter Developper API, instead of using the jar file")
    parser.add_argument("--tfidf_model_path", type=str, default='../data/2013to2016_tfidf_vectorizer_20190109.pkl')
    parser.add_argument("--fasttext_model_path", type=str, default='../data/text_sample_2013to2016_gensim_200.model')
    parser.add_argument("--glove_path", type=str, default='../data/glove.twitter.27B.200d.txt')
    # Features
    parser.add_argument("--fasttext_merge", type=str, default='avg',
                        help='avg | weighted')
    parser.add_argument("--glove_merge", type=str, default='avg',
                        help='avg | weighted')
    # Model
    parser.add_argument("--model", type=str, default='bernoulli_nb',
                        help="bernoulli_nb | rf | sgd_svm")
    parser.add_argument("--no_class_weight", action='store_true',
                        help="Not use different weights for each class")
    parser.add_argument("--cv_num", type=int, default=5,
                        help="The n-fold cross-validation")
    return parser.parse_args()
