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
    parser.add_argument("--search_print_interval", type=int, default=3,
                        help="As random forest has much longer training time, suggested value is 3, and nb can use 20")
    parser.add_argument("--search_skip", type=int, default=0,
                        help="Skip some search (because some parameters have been searched in previous running)")
    parser.add_argument("--use_stratify_split", action="store_true",
                        help="Use stratify split for multi-label setting implemented based on a 2011 paper")
    parser.add_argument("--predict_mode", action="store_true",
                        help="In this mode, write the dev_predict, dev_label, test_predict to file, for later ensemble")
    parser.add_argument("--ensemble", type=str, default=None,
                        help="After running all models with --predict_mode, you can run with it to ensemble all models"
                             "Can choose from voting | svm_linear | svm_rbf | logistic_reg")
    parser.add_argument("--force_retrain", action="store_true",
                        help="force to retrain the model on 2018 data")
    # Regression for priority score
    parser.add_argument("--train_regression", action="store_true",
                        help="Train regressor on the priority score, and override the score for the final submission.")
    parser.add_argument("--merge_priority_score", type=str, default='simple',
                        help="simple | advanced, simple means use all regression or use all score from class prediction"
                        " and advanced means merging those two sources by some weights.")
    parser.add_argument("--advanced_predict_weight", type=float, default=0.5,
                        help="The weight to merge prediction and regression result in the advanced method.")
    # Data path
    parser.add_argument("--out_dir", type=str, default='out',
                        help="Directory contains the output things, including the log and ckpt")
    parser.add_argument("--test_dir", type=str, default='test',
                        help="Directory contains the small dataset for testing")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="Directory contains the data")
    parser.add_argument("--log_name", type=str, default='defaultLog',
                        help="Name of logfile, could use Date such as Apr25")
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
                        choices=['bernoulli_nb', 'rf', 'sgd_svm', 'svm_linear', 'xgboost'],
                        help="bernoulli_nb | rf | sgd_svm | svm_linear | xgboost")
    parser.add_argument("--class_weight_scheme", type=str, default='balanced', choices=['balanced', 'customize'],
                        help="balanced | customize, if set to customize, we use the weight calculated by training file")
    parser.add_argument("--additional_weight", type=float, default=0.2,
                        help="Additional weights added to the actionable classes (defined by official host), and"
                             "it is only used when class_weight_scheme=customize.")
    parser.add_argument("--cv_num", type=int, default=5,
                        help="The n-fold cross-validation")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Can set to -1 to use all cpu cores")
    # For final submission
    parser.add_argument("--get_submission", action="store_true",
                        help="Generate submission file for TREC-IS")
    parser.add_argument("--pick_criteria", type=str, default='top', choices=['top', 'threshold', 'autothre'],
                        help='for threshold you can set pick_threshold, and for top you can set pick_k, '
                             'for autothre you don\'t need to set anything, because it uses different threshold for each class')
    parser.add_argument("--pick_threshold", type=float, default=None,
                        help='If None, our model will search best threshold')
    parser.add_argument("--pick_k", type=int, default=2,
                        help='Pick the top k for each prediction')

    return parser.parse_args()
