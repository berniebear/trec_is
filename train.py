import os
from typing import List, Dict
from scipy import stats
import numpy as np
import pandas as pd
import scipy
import pickle
from sklearn.model_selection import KFold, RandomizedSearchCV, ParameterSampler, GridSearchCV, ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.utils import shuffle
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from utils import print_to_log, anytype_f1_scorer
import utils


class Train(object):
    def __init__(self, args, data_x: np.ndarray, data_y: np.ndarray, id2label: list,
                 feature_lens: List[int], class_weight: List[float],
                 event_type: str=None):
        """
        We use the feature_lens to make the API consistent with or without late fusion.
        If we use late fusion, the feature_lens will contain real lens of different features.
        If we don't use late fusion, feature_lens will contain a fake number which is the len of concatenated feature
        Please see details in _fit_data and _predict_data about how we cope with late fusion
        :param args:
        :param data_x:
        :param data_y:
        :param id2label:
        :param feature_lens: Contains lens of features, such as [1024, 256] means the first feature is 1024 dim and the
                second feature is 256 dim
        :param class_weight: Weights of each class calculated by `get_class_weight` in utils.py
        :param event_type: used to specify the event for event-wise model
        """
        self.args = args
        self.data_x = data_x
        self.data_y = data_y
        self.id2label = id2label
        self.feature_lens = feature_lens
        self.metric_names: List[str] = ['accuracy', 'precision', 'recall', 'f1'] if args.class_weight_scheme == 'balanced' else ['weighted_ce']
        self.clf: List[OneVsRestClassifier] = None
        self.event_type = event_type

        self.class_weight_list = class_weight
        if self.args.class_weight_scheme == 'balanced':
            self.class_weight = 'balanced'
        else:
            self.class_weight = {i: weight for i, weight in enumerate(class_weight)}

    def train_on_all(self):
        """
        A wrapper for train on all data, which is used to prepare for the prediction on test data
        Notice that here we don't use cross-validation, because cv is only used for parameter-choosing
        Now we have determined the parameter, and we want to train on all data we have (self.data_x and self.data_y)
        :return:
        """
        custom_postfix = '_{}'.format(self.event_type) if self.event_type is not None else ''
        model_save_name = '{0}_{1}.pkl'.format(self.args.model, custom_postfix)
        ckpt_file = os.path.join(self.args.model_dir, model_save_name)
        if os.path.isfile(ckpt_file) and not self.args.force_retrain:
            with open(ckpt_file, 'rb') as f:
                self.clf = pickle.load(f)
        else:
            self._create_model()
            self._binarize_data_y()
            self._fit_data(self.data_x, self.data_y)
            with open(ckpt_file, 'wb') as f:
                pickle.dump(self.clf, f)

    def _fit_data(self, data_x, data_y):
        start_idx = 0
        for i_clf, feat_len in enumerate(self.feature_lens):
            end_idx = start_idx + feat_len
            self.clf[i_clf].fit(data_x[:, start_idx: end_idx], data_y)
            start_idx = end_idx

    def _predict_data(self, data_x):
        """
        Currently support avg score or voting
        Notice that this is a uniform API for both with or without late fusion.
            For late fusion it will use avg or voting
            For not late fusion it will have the same effect as the single model prediction
        :param data_x:
        :return:
        """
        return self._predict_data_by_voting(data_x)

    def _predict_data_by_avg_score(self, data_x):
        """
        Todo: Now we use simple average, but can use more sophisticated method such as weighted average and so on
        :param data_x:
        :return:
        """
        start_idx = 0
        predict_score = np.zeros([data_x.shape[0], len(self.id2label)], dtype=np.float64)
        for i_clf, feat_len in enumerate(self.feature_lens):
            end_idx = start_idx + feat_len
            predict_score += self.clf[i_clf].predict_proba(data_x[:, start_idx: end_idx])
            start_idx = end_idx
        return np.argmax(predict_score, axis=-1)

    def _predict_data_by_voting(self, data_x):
        """
        Todo: A more sophisticated method is to set weight according to their performance on a hold-out dataset
        :param data_x:
        :return:
        """
        start_idx = 0
        labels_count = [dict() for i in range(data_x.shape[0])]
        for i_clf, feat_len in enumerate(self.feature_lens):
            end_idx = start_idx + feat_len
            predict_score = self.clf[i_clf].predict_proba(data_x[:, start_idx: end_idx])
            predict_label = np.argmax(predict_score, axis=-1)
            for i_idx, it_label in enumerate(predict_label):
                labels_count[i_idx][it_label] = labels_count[i_idx].get(it_label, 0) + 1
            start_idx = end_idx
        vote_res = [0] * data_x.shape[0]
        for i, count_dict in enumerate(labels_count):
            max_vote = -1
            for predict, count in count_dict.items():
                if count > max_vote:
                    max_vote = count
                    vote_res[i] = predict
        return np.asarray(vote_res)

    def _binarize_data_y(self):
        self.mlb = MultiLabelBinarizer(classes=list(range(len(self.id2label))))
        self.data_y = self.mlb.fit_transform(self.data_y)

    def train(self):
        self._create_model()
        self._binarize_data_y()
        if self.args.search_best_parameters:
            self._random_search_best_para(self.args.random_search_n_iter)
        if self.args.cross_validate:
            return self._cross_validate()
        self._fit_data(self.data_x, self.data_y)

    def predict(self, data_x: np.ndarray, tweetid_list: list, tweetid2idx: list, tweetid2incident: dict,
                id2label: list, short2long_label: dict, majority_label: str, out_file: str):
        """
        For those missed tweetid (that cannot be found in twitter API), we use the majority label as the prediction res.
        As we can see in the evaluation script, the rank filed doesn't matter.
        Todo: Currently we only care about the category prediction, and we don't care about the score, but we need to
            care about the "priority" in 2019-A task
        :param data_x: Feature of data
        :param tweetid_list:
        :param tweetid2idx: Can find the actuall idx of this tweetid in data_x
        :param tweetid2incident:
        :param id2label:
        :param short2long_label: the output format need the long label in the form of A-B
        :param majority_label:
        :param out_file:
        :return:
        """
        fout = open(out_file, 'w', encoding='utf8')
        predict_res = self._predict_data(data_x)
        count_label = []
        for tweetid in tweetid_list:
            incident = tweetid2incident[tweetid]
            label = id2label[predict_res[tweetid2idx[tweetid]]] if tweetid in tweetid2idx else majority_label
            label = short2long_label[label]
            fout.write("{0}\tQ0\t{1}\t1\t1.0\t{2}\tmyrun\n".format(incident, tweetid, label))
            count_label.append({"tweet_id": tweetid, "label": label})
        fout.close()
        df = pd.DataFrame(count_label)
        print_to_log("{} rows have been replaced due to missing of tweetid".format(len(tweetid_list) - len(tweetid2idx)))
        print_to_log("The count of different labels in prediction results:\n{}".format(df.groupby("label").count()))
        print_to_log("The prediction file has been written to {}".format(out_file))

    def shuffle_data(self):
        self.data_x, self.data_y = shuffle(self.data_x, self.data_y)

    def _create_model(self, param=None):
        self.clf = [self._create_single_model(param) for i in range(len(self.feature_lens))]

    def _create_single_model(self, param=None):
        model_name = self.args.model
        print_to_log("The model used here is {0}".format(model_name))
        if model_name == 'sgd_svm':
            clf = SGDClassifier(max_iter=1000, tol=1e-3, loss='hinge', class_weight=self.class_weight)
        elif model_name == 'svm_linear':
            if not param:
                param = {'class_weight': self.class_weight, "C": 0.1, "dual": False, "penalty": "l2"}
            clf = CalibratedClassifierCV(LinearSVC(**param))  # Set dual=False when training num >> feature num
        elif model_name == 'svm_rbf':
            clf = SVC(kernel='rbf', class_weight=self.class_weight, gamma='auto', probability=True)
        elif model_name == 'svm_rbf_scale':
            clf = SVC(kernel='rbf', class_weight=self.class_weight, gamma='scale', probability=True)
        elif model_name == 'svm_chi2':
            clf = SVC(kernel=chi2_kernel, class_weight=self.class_weight, probability=True)
        elif model_name == 'gs_nb':
            clf = GaussianNB()
        elif model_name == 'bernoulli_nb':
            if not param:
                if self.args.class_weight_scheme == 'balanced':
                    param = {'alpha': 0.8490, 'binarize': 0.3086, 'fit_prior': True}
                else:
                    param = {'alpha': 0.4974, 'binarize': 0.7751, 'fit_prior': True}
            clf = BernoulliNB(**param)
        elif model_name == 'rf':
            if not param:
                param = {
                    'n_estimators': 128,
                    "n_jobs": 1,
                    'class_weight': self.class_weight,
                    'criterion': 'gini',
                    'max_depth': 64,
                    'max_features': 213,
                    'min_samples_leaf': 5,
                    'min_samples_split': 43,
                }
            clf = RandomForestClassifier(**param)
        elif model_name == 'xgboost':
            if not param:
                if self.args.class_weight_scheme == 'balanced':
                    param = {'subsample': 0.9,
                             'n_jobs': 1,
                             'n_estimators': 500,
                             'max_depth': 8,
                             'learning_rate': 0.05,
                             'gamma': 0,
                             'colsample_bytree': 0.9,
                             }
                else:
                    param = {'subsample': 0.9,
                             'n_jobs': 1,
                             'n_estimators': 300,
                             'max_depth': 3,
                             'learning_rate': 0.03,
                             'gamma': 5,
                             'colsample_bytree': 1.0,
                             }
            clf = XGBClassifier(**param)
        else:
            raise NotImplementedError
        return OneVsRestClassifier(clf, n_jobs=self.args.n_jobs)

    def _random_search_best_para(self, n_iter):
        if self.args.search_by_sklearn_api:
            self._search_by_sklearn(n_iter)
        else:
            self._search_by_our_own(n_iter)

    def _search_by_sklearn(self, n_iter):
        """
        Use the RandomizedSearchCV API of sklearn, but need to customize the scoring function
        The advantage is that it parallelized well (However, according to the warning
            "Multiprocessing-backed parallel loops cannot be nested", if the model is parallelized,
            the random search will be serielized automatically)
        Notice that as the model clf is stored as an attribute named estimator inside the OneVsRestClassifier model,
            we should add "estimator__" as prefix for setting their parameters in the OneVsRestClassifier wrapper
        Another thing to notice is that Because parallel jobs cannot be nested, we can set model to be paralled and
            search to be sequential, or model to be sequential but search to be parallel.
        :param n_iter:
        :return:
        """
        if self.args.model == 'rf':
            clf = RandomForestClassifier(n_estimators=128, class_weight=self.class_weight, n_jobs=1)
            param_dist = {
                "estimator__max_depth": [2, 4, 8, 16, 32, 64, 128, None],
                "estimator__max_features": scipy.stats.randint(1, 512),
                "estimator__min_samples_split": scipy.stats.randint(2, 512),
                "estimator__min_samples_leaf": scipy.stats.randint(2, 512),
                "estimator__criterion": ["gini", "entropy"],
            }
        elif self.args.model == 'bernoulli_nb':
            clf = BernoulliNB()
            param_dist = {
                "estimator__alpha": scipy.stats.uniform(),
                "estimator__binarize": scipy.stats.uniform(),
                "estimator__fit_prior": [True, False],
            }
        elif self.args.model == 'svm_linear':
            clf = CalibratedClassifierCV(LinearSVC())
            param_dist = {
                "penalty": ['l1', 'l2'],
                "C": [0.1, 1, 10, 100, 1000],
                "class_weight": [self.class_weight],
                "dual": [False],
            }
        elif self.args.model == 'xgboost':
            clf = XGBClassifier()
            param_dist = {
                "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
                "n_estimators": [100, 300, 500],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 1, 5],
                "n_jobs": [1],
            }
        else:
            raise ValueError("The model {} doesn't support parameter search in current stage".format(self.args.model))

        param_dist = {"estimator__{}".format(k): v for k, v in param_dist.items()}
        clf = OneVsRestClassifier(clf, n_jobs=1)
        kf = KFold(n_splits=self.args.cv_num, random_state=self.args.random_seed)
        # Notice that as we use clf.predict_proba in our cross-validation, we need to set needs_proba=True here
        scorer = make_scorer(anytype_f1_scorer, greater_is_better=True, needs_proba=True, id2label=self.id2label)
        if self.args.model == 'svm_linear':
            search = GridSearchCV(clf, param_grid=param_dist, cv=kf, scoring=scorer, n_jobs=-1, verbose=10)
        else:
            search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter, cv=kf,
                                        scoring=scorer, n_jobs=-1, verbose=10)

        search.fit(self.data_x, self.data_y)

        print_to_log("Random Search finished!")
        print_to_log("best_score_:\n{}".format(search.best_score_))
        print_to_log("best_params_:\n{}".format(search.best_params_))
        quit()

    def _search_by_our_own(self, n_iter):
        """
        Call our own class method to perform the random search
        The drawback is that they cannot be performed paralleled
        :param n_iter:
        :return:
        """
        if self.args.model == 'rf':
            param_dist = {
                "max_depth": [2, 4, 8, 16, 32, 64, 128, None],
                "max_features": scipy.stats.randint(1, 512),
                "min_samples_split": scipy.stats.randint(2, 512),
                "min_samples_leaf": scipy.stats.randint(2, 512),
                "criterion": ["gini", "entropy"],
                "n_estimators": [128],
                "class_weight": [self.class_weight],
                "n_jobs": [1],
            }
        elif self.args.model == 'bernoulli_nb':
            param_dist = {
                "alpha": scipy.stats.uniform(),
                "binarize": scipy.stats.uniform(),
                "fit_prior": [True, False],
            }
        elif self.args.model == 'svm_linear':
            param_dist = {
                "penalty": ['l1', 'l2'],
                "C": [0.1, 1, 10, 100, 1000],
                "class_weight": [self.class_weight],
                "dual": [False],
            }
        elif self.args.model == 'xgboost':
            param_dist = {
                "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
                "n_estimators": [100, 300, 500],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 1, 5],
                "n_jobs": [1],
            }
        else:
            raise ValueError("The model {} doesn't support parameter search in current stage".format(self.args.model))

        if self.args.model == 'svm_linear':
            param_list = list(ParameterGrid(param_dist))
        else:
            param_list = list(ParameterSampler(param_dist, n_iter=n_iter))

        metric_name = 'f1' if self.args.class_weight_scheme == 'balanced' else 'weighted_ce'
        best_metric = float("-inf")
        best_param = dict()
        for i, param in enumerate(param_list):
            if i < self.args.search_skip:
                continue
            print_to_log("Using the parameter set: {}".format(param))
            self._create_model(param)
            current_metric = self._cross_validate()
            if current_metric > best_metric:
                best_metric = current_metric
                best_param = param
            if (i + 1) % self.args.search_print_interval == 0:
                print_to_log("After searching {0} sets of parameters, current best is {1}, best {3} is {2}".format(
                    i + 1, best_param, best_metric, metric_name))

        print_to_log("The Random search finished!")
        print_to_log("The best {0} is {1}".format(metric_name, best_metric))
        print_to_log("The best parameter is {}".format(best_param))
        quit()

    def _simple_cross_validate(self):
        """
        Use a simple fixed NB model to double check the correctness of sklearn Random search and my random search
        It can confirm our API compatible with late-fusion is correct
        :return:
        """
        kf = KFold(n_splits=self.args.cv_num, random_state=self.args.random_seed)
        metric_values = {metric_name: [] for metric_name in self.metric_names}
        clf = BernoulliNB(alpha=0.8490, binarize=0.3086, fit_prior=True)
        clf = OneVsRestClassifier(clf, n_jobs=self.args.n_jobs)
        for train_idx_list, test_idx_list in kf.split(self.data_x, self.data_y):
            X_train = self.data_x[train_idx_list]
            y_train = self.data_y[train_idx_list]
            X_test = self.data_x[test_idx_list]
            y_test = self.data_y[test_idx_list]
            clf.fit(X_train, y_train)
            y_predict_score = clf.predict_proba(X_test)
            y_predict = np.argmax(y_predict_score, axis=-1)
            metric_results = utils.evaluate_any_type(y_test, y_predict, self.id2label)
            for metric_name in self.metric_names:
                metric_values[metric_name].append([metric_results[metric_name], len(y_test)])

        metric_weighted_avg = self._get_weighted_avg(metric_values)
        for metric_name in ['f1']:
            print_to_log('The {0} score in cross validation is {1}'.format(metric_name, metric_values[metric_name]))
            print_to_log('The average {0} score is {1}'.format(metric_name, metric_weighted_avg[metric_name]))
        quit()

    def _get_k_fold_index_list(self):
        """
        We can simply use the KFold API provided by sklearn, or use the stratified split designed for multi-label
        To make it consistent with the sklearn API, we do some post-processing
        :return:
        """
        if self.args.use_stratify_split:
            index_list = utils.get_k_fold_index_list(self.data_y, self.id2label, self.args.cv_num)
        else:
            # StratifiedKFold doesn't support multi-label setting, so we can only use KFold
            kf = KFold(n_splits=self.args.cv_num, random_state=self.args.random_seed, shuffle=True)
            index_list = kf.split(self.data_x, self.data_y)

        return index_list

    def _get_label_count(self, y_data: np.ndarray):
        y_count = {i: 0 for i in range(len(self.id2label))}
        for i in range(y_data.shape[0]):
            for j in range(y_data.shape[1]):
                if y_data[i, j] == 1:
                    y_count[j] += 1
        return y_count

    def _get_ratio_for_each_class(self, y_train: np.ndarray, y_test: np.ndarray):
        """
        This method is used to compare different k-fold stratified sampling method. For example, if you use 5-fold,
            the method would be better if its ratio is closer to 4.0
        You can call this function in `_cross_validate` method
        :param y_train:
        :param y_test:
        :return:
        """
        y_train_count = self._get_label_count(y_train)
        y_test_count = self._get_label_count(y_test)
        count_ratio = {i: y_train_count[i]/y_test_count[i] if y_test_count[i] != 0 else 0 for i in range(len(self.id2label))}
        sum_val = 0
        sum_count = 0
        for i in range(len(self.id2label)):
            current_count = y_train_count[i] + y_test_count[i]
            sum_val += count_ratio[i] * current_count
            sum_count += current_count
        print("The ratio for each class is {}".format(count_ratio))
        print("The weighted average ratio is {}".format(sum_val / sum_count))

    def _cross_validate(self):
        """
        If we are performing event-wise training, we need to return the metrics for each running (event)
        Note: If you want to get more balanced k-fold split, you can refer to `proba_mass_split` in utils.py,
            or the `stratify_split` in utils.py which is implemented based on Sechidis et. al paper
        :return:
        """
        print_to_log('Use {} fold cross validation'.format(self.args.cv_num))
        metric_values = {metric_name: [] for metric_name in self.metric_names}
        dev_predict = np.zeros_like(self.data_y, dtype=np.float)

        index_list = self._get_k_fold_index_list()
        for train_idx_list, test_idx_list in index_list:
            X_train = self.data_x[train_idx_list]
            y_train = self.data_y[train_idx_list]
            X_test = self.data_x[test_idx_list]
            y_test = self.data_y[test_idx_list]
            self._fit_data(X_train, y_train)
            predict_score = self.clf[0].predict_proba(X_test)
            dev_predict[test_idx_list] = predict_score

            if self.args.class_weight_scheme == 'balanced':
                y_predict = self._predict_data(X_test)
                metric_results = utils.evaluate_any_type(y_test, y_predict, self.id2label)
            else:
                metric_results = utils.evaluate_weighted_sum(y_test, predict_score, self.class_weight_list)

            for metric_name in self.metric_names:
                metric_values[metric_name].append([metric_results[metric_name], len(y_test)])

        metric_weighted_avg = self._get_weighted_avg(metric_values)
        for metric_name in self.metric_names:
            print_to_log('The {0} score in cross validation is {1}'.format(metric_name, metric_values[metric_name]))
            print_to_log('The average {0} score is {1}'.format(metric_name, metric_weighted_avg[metric_name]))

        if self.args.search_best_parameters:
            target_metric = 'f1' if self.args.class_weight_scheme == 'balanced' else 'weighted_ce'
            return metric_weighted_avg[target_metric]

        return {metric_name: metric_weighted_avg[metric_name] for metric_name in self.metric_names}, dev_predict

    def _get_weighted_avg(self, metric_values):
        return utils.get_weighted_avg(metric_values, self.metric_names)

    def predict_on_test(self, test_data: np.ndarray):
        """
        For ensemble purpose, each model predict on test data
        Notice that for the event-wise model, we need to merge all predictions into a list to keep the original order,
            which has been done in main.py
        :param test_data:
        :return:
        """
        return self.clf[0].predict_proba(test_data)
