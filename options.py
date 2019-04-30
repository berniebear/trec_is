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
    parser.add_argument("--search_by_sklearn_api", action='store_true',
                        help="Use the sklearn API (GridSearchCV and RandomSearchCV)")
    parser.add_argument("--search_print_interval", type=int, default=20,
                        help="As random forest has much longer training time, suggested value is 3, and nb can use 20")
    parser.add_argument("--search_skip", type=int, default=0,
                        help="Skip some search (because some parameters have been searched in previous running)")
    parser.add_argument("--use_stratify_split", action="store_true",
                        help="Use stratify split for multi-label setting implemented based on a 2011 paper")
    parser.add_argument("--predict_mode", action="store_true",
                        help="In this mode, write the dev_predict, dev_label, test_predict to file, for later ensemble")
    parser.add_argument("--ensemble", action='store_true',
                        help="After running all models with --predict_mode, you can run with it to ensemble all models")
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
    # Features
    parser.add_argument("--fasttext_merge", type=str, default='avg',
                        help='avg | weighted')
    parser.add_argument("--glove_merge", type=str, default='avg',
                        help='avg | weighted')
    parser.add_argument("--use_pca", action="store_true",
                        help="Use PCA to reduce dimension of each kind of feature")
    parser.add_argument("--pca_dim", type=int, default=100,
                        help="Dimension after using PCA")
    parser.add_argument("--normalize_feat", action="store_true",
                        help="Normalize each feature when reading them from file")
    # Model
    parser.add_argument("--model", type=str, default='bernoulli_nb',
                        help="bernoulli_nb | rf | sgd_svm | svm_linear | xgboost")
    parser.add_argument("--no_class_weight", action='store_true',
                        help="Not use different weights for each class")
    parser.add_argument("--cv_num", type=int, default=5,
                        help="The n-fold cross-validation")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Can set to -1 to use all cpu cores")
    return parser.parse_args()
