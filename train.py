from typing import List, Dict
from scipy import stats
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import KFold, RandomizedSearchCV, ParameterSampler, GridSearchCV, ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.utils import shuffle
from xgboost import XGBClassifier

from utils import print_to_log, anytype_f1_scorer
import utils


class Train(object):
    def __init__(self, args, data_x: np.ndarray, data_y: np.ndarray, id2label: list, feature_lens: List[int]):
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
        """
        self.args = args
        self.data_x = data_x
        self.data_y = data_y
        self.id2label = id2label
        self.feature_lens = feature_lens
        self.metric_names = ['accuracy', 'precision', 'recall', 'f1']
        self.clf: List[OneVsRestClassifier] = None

    def _fit_data(self, data_x, data_y):
        start_idx = 0
        for i_clf, feat_len in enumerate(self.feature_lens):
            end_idx = start_idx + feat_len
            self.clf[i_clf].fit(data_x[:, start_idx: end_idx], data_y)
            start_idx = end_idx

    def _predict_data(self, data_x):
        """
        Currently support avg score or voting
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
            if 'svm' in self.args.model:
                predict_score += self.clf[i_clf].decision_function(data_x[:, start_idx: end_idx])
            else:
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
            if 'svm' in self.args.model:
                predict_score = self.clf[i_clf].decision_function(data_x[:, start_idx: end_idx])
            else:
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

    def train(self, small_x=None, small_y=None):
        self._shuffle_data()
        self._create_model()
        self.data_y = MultiLabelBinarizer(classes=list(range(len(self.id2label)))).fit_transform(self.data_y)
        if self.args.search_best_parameters:
            self._random_search_best_para(self.args.random_search_n_iter)
        if self.args.train_on_small:
            self._train_on_small_predict_on_large(small_x, small_y)
        if self.args.cross_validate:
            return self._cross_validate()
        self._fit_data(self.data_x, self.data_y)

    def _train_on_small_predict_on_large(self, small_x, small_y):
        print("Train on the small and test on cross-validate test")
        small_y = MultiLabelBinarizer(classes=list(range(len(self.id2label)))).fit_transform(small_y)
        self._fit_data(small_x, small_y)
        kf = KFold(n_splits=self.args.cv_num)
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        metric_values = {metric_name: [] for metric_name in metric_names}
        for train, test in kf.split(self.data_x, self.data_y):
            X_test = self.data_x[test]
            y_test = self.data_y[test]
            y_predict = self._predict_data(X_test)
            metric_results = utils.evaluate_any_type(y_test, y_predict, self.id2label)
            for metric_name in metric_names:
                metric_values[metric_name].append(metric_results[metric_name])
        for metric_name in metric_names:
            print('The {0} score in cross validation is {1}'.format(metric_name, metric_values[metric_name]))
            print('The average {0} score is {1}'.format(metric_name, np.mean(metric_values[metric_name])))
        quit()

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

    def _shuffle_data(self):
        self.data_x, self.data_y = shuffle(self.data_x, self.data_y)

    def _create_model(self, param=None):
        self.clf = [self._create_single_model(param) for i in range(len(self.feature_lens))]

    def _create_single_model(self, param=None):
        class_weight = None if self.args.no_class_weight else 'balanced'
        model_name = self.args.model
        print_to_log("The model used here is {0}".format(model_name))
        if model_name == 'sgd_svm':
            clf = SGDClassifier(max_iter=1000, tol=1e-3, loss='hinge', class_weight=class_weight)
        elif model_name == 'svm_linear':
            if not param:
                param = {'class_weight': class_weight, "C": 0.1, "dual": False, "penalty": "l2"}
            clf = LinearSVC(**param)  # Set dual=False when training num >> feature num
        elif model_name == 'svm_rbf':
            clf = SVC(kernel='rbf', class_weight=class_weight, gamma='auto')
        elif model_name == 'svm_rbf_scale':
            clf = SVC(kernel='rbf', class_weight=class_weight, gamma='scale')
        elif model_name == 'svm_chi2':
            clf = SVC(kernel=chi2_kernel, class_weight=class_weight)
        elif model_name == 'gs_nb':
            clf = GaussianNB()
        elif model_name == 'bernoulli_nb':
            if not param:
                param = {'alpha': 0.8490, 'binarize': 0.3086, 'fit_prior': True}
            clf = BernoulliNB(**param)
        elif model_name == 'rf':
            if not param:
                param = {
                    'n_estimators': 128,
                    "n_jobs": -1,
                    'class_weight': class_weight,
                    'criterion': 'gini',
                    'max_depth': 32,
                    'max_features': 113,
                    'min_samples_leaf': 2,
                    'min_samples_split': 54,
                }
            clf = RandomForestClassifier(**param)
        elif model_name == 'xgboost':
            clf = XGBClassifier()
        else:
            raise NotImplementedError
        return OneVsRestClassifier(clf, n_jobs=-1)

    def _random_search_best_para(self, n_iter):
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
            clf = RandomForestClassifier(n_estimators=128, class_weight="balanced", n_jobs=1)
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
            clf = LinearSVC(class_weight='balanced', dual=False)
            param_dist = {
                "estimator__penalty": ['l1', 'l2'],
                "estimator__C": [0.1, 1, 10, 100, 1000]
            }
        else:
            raise ValueError("The model {} doesn't support parameter search in current stage".format(self.args.model))

        clf = OneVsRestClassifier(clf, n_jobs=1)
        kf = KFold(n_splits=self.args.cv_num, random_state=self.args.random_seed)
        # Notice that as we use clf.predict_proba in our cross-validation, we need to set needs_proba=True here
        scorer = make_scorer(anytype_f1_scorer, greater_is_better=True, needs_proba=True, id2label=self.id2label)
        if self.args.model == 'svm_linear':
            search = GridSearchCV(clf, param_grid=param_dist, cv=kf, scoring=scorer, n_jobs=-1)
        else:
            search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter, cv=kf, scoring=scorer, n_jobs=-1)

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
                "class_weight": ['balanced'],
                "dual": [False],
            }
        elif self.args.model == 'xgboost':
            param_dist = {
                "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
                "n_estimators": [100, 300, 500],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 1, 5]
            }
        else:
            raise ValueError("The model {} doesn't support parameter search in current stage".format(self.args.model))

        if self.args.model == 'svm_linear':
            param_list = list(ParameterGrid(param_dist))
        else:
            param_list = list(ParameterSampler(param_dist, n_iter=n_iter))

        best_f1 = 0.0
        best_param = dict()
        for i, param in enumerate(param_list):
            if self.args.model == 'rf':  # Some fix parameters of random forest
                param.update({'n_estimators': 128, 'class_weight': "balanced", 'n_jobs': -1})
            self._create_model(param)
            current_f1 = self._cross_validate()
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_param = param
            if (i + 1) % self.args.search_print_interval == 0:
                print_to_log("After searching {0} sets of parameters, current best is {1}, best F1 is {2}".format(
                    i + 1, best_param, best_f1))

        print_to_log("The Random search finished!")
        print_to_log("The best f1 is {}".format(best_f1))
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
        clf = OneVsRestClassifier(clf, n_jobs=-1)
        for train, test in kf.split(self.data_x, self.data_y):
            X_train = self.data_x[train]
            y_train = self.data_y[train]
            X_test = self.data_x[test]
            y_test = self.data_y[test]
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
            stratified_data_ids, _ = utils.stratify_split(self.data_y, list(range(len(self.id2label))),
                                                          [1 / self.args.cv_num] * self.args.cv_num, one_hot=True)
            index_list = []
            for i in range(self.args.cv_num):
                test_idx = stratified_data_ids[i]
                train_idx = []
                for ii in range(self.args.cv_num):
                    if ii == i:
                        continue
                    train_idx += stratified_data_ids[ii]
                index_list.append((train_idx, test_idx))

        else:
            # StratifiedKFold doesn't support multi-label setting, so we can only use KFold
            kf = KFold(n_splits=self.args.cv_num, random_state=self.args.random_seed)
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
        Don't worry about stratified K-fold, because for cross_validate,
            if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used
        If we are performing event-wise training, we need to return the metrics for each running (event)
        Note: If you want to get more balanced k-fold split, you can refer to `proba_mass_split` in utils.py,
            or the `stratify_split` in utils.py which is implemented based on Sechidis et. al paper
        :return:
        """
        print_to_log('Use {} fold cross validation'.format(self.args.cv_num))
        metric_values = {metric_name: [] for metric_name in self.metric_names}

        index_list = self._get_k_fold_index_list()
        for train, test in index_list:
            X_train = self.data_x[train]
            y_train = self.data_y[train]
            X_test = self.data_x[test]
            y_test = self.data_y[test]
            self._fit_data(X_train, y_train)
            y_predict = self._predict_data(X_test)
            metric_results = utils.evaluate_any_type(y_test, y_predict, self.id2label)
            for metric_name in self.metric_names:
                metric_values[metric_name].append([metric_results[metric_name], len(y_test)])

        metric_weighted_avg = self._get_weighted_avg(metric_values)
        for metric_name in self.metric_names:
            print_to_log('The {0} score in cross validation is {1}'.format(metric_name, metric_values[metric_name]))
            print_to_log('The average {0} score is {1}'.format(metric_name, metric_weighted_avg[metric_name]))

        if self.args.event_wise:
            return {metric_name: metric_weighted_avg[metric_name] for metric_name in self.metric_names}
        elif self.args.search_best_parameters:
            return metric_weighted_avg['f1']
        else:
            quit()

    def _get_weighted_avg(self, metric_values):
        metric_accumulate = {metric_name: 0.0 for metric_name in self.metric_names}
        count = {metric_name: 0 for metric_name in self.metric_names}
        for metric_name in self.metric_names:
            for value, length in metric_values[metric_name]:
                metric_accumulate[metric_name] += value * length
                count[metric_name] += length
        for metric_name in self.metric_names:
            metric_accumulate[metric_name] /= count[metric_name]
        return metric_accumulate
