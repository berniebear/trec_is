from typing import List
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils import shuffle

from utils import print_to_log
import utils


class Train(object):
    def __init__(self, args, data_x: np.ndarray, data_y: np.ndarray, id2label: list, feature_lens: List[int]):
        """
        We use the feature_lens to make the API consistent with or withnot late fusion.
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
        self.data_y = MultiLabelBinarizer(classes=[i for i in range(len(self.id2label))]).fit_transform(self.data_y)
        if self.args.train_on_small:
            self._train_on_small_predict_on_large(small_x, small_y)
        if self.args.cross_validate:
            self._cross_validate()
        self._fit_data(self.data_x, self.data_y)

    def _train_on_small_predict_on_large(self, small_x, small_y):
        print("Train on the small and test on cross-validate test")
        small_y = MultiLabelBinarizer(classes=[i for i in range(len(self.id2label))]).fit_transform(small_y)
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

    def _create_model(self):
        self.clf = [self._create_single_model() for i in range(len(self.feature_lens))]

    def _create_single_model(self):
        class_weight = None if self.args.no_class_weight else 'balanced'
        model_name = self.args.model
        print_to_log("The model used here is {0}".format(model_name))
        if model_name == 'sgd_svm':
            clf = SGDClassifier(max_iter=1000, tol=1e-3, loss='hinge', class_weight=class_weight)
        elif model_name == 'svm_linear':
            clf = LinearSVC(class_weight=class_weight, dual=True)  # Feature dim is large, should use dual
        elif model_name == 'svm_rbf':
            clf = SVC(kernel='rbf', class_weight=class_weight, gamma='auto')
        elif model_name == 'svm_rbf_scale':
            clf = SVC(kernel='rbf', class_weight=class_weight, gamma='scale')
        elif model_name == 'svm_chi2':
            clf = SVC(kernel=chi2_kernel, class_weight=class_weight)
        elif model_name == 'bernoulli_nb':
            clf = BernoulliNB(alpha=0.0158, binarize=0.7317, fit_prior=False)
        elif model_name == 'rf':
            rf_params = {
                'n_estimators': 128,
                "n_jobs": -1,
                'class_weight': class_weight,
                'criterion': 'gini',
                'max_depth': 32,
                'max_features': 113,
                'min_samples_leaf': 2,
                'min_samples_split': 54,
            }
            clf = RandomForestClassifier(**rf_params)
        else:
            raise NotImplementedError
        return OneVsRestClassifier(clf)

    def _cross_validate(self):
        """
        Don't worry about stratified K-fold, because for cross_validate,
            if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used
        Todo: If you want to get more balanced k-fold split, you can refer to `proba_mass_split` in utils.py
        :return:
        """
        print_to_log('Use {} fold cross validation'.format(self.args.cv_num))
        kf = KFold(n_splits=self.args.cv_num)  # As StratifiedKFold doesn't support multi-label setting
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        metric_values = {metric_name: [] for metric_name in metric_names}
        for train, test in kf.split(self.data_x, self.data_y):
            X_train = self.data_x[train]
            y_train = self.data_y[train]
            X_test = self.data_x[test]
            y_test = self.data_y[test]
            self._fit_data(X_train, y_train)
            y_predict = self._predict_data(X_test)
            metric_results = utils.evaluate_any_type(y_test, y_predict, self.id2label)
            for metric_name in metric_names:
                metric_values[metric_name].append(metric_results[metric_name])
        for metric_name in metric_names:
            print_to_log('The {0} score in cross validation is {1}'.format(metric_name, metric_values[metric_name]))
            print_to_log('The average {0} score is {1}'.format(metric_name, np.mean(metric_values[metric_name])))
        quit()

    def get_best_hyper_parameter(self, n_iter_search=100, r_state=1337):
        if self.args.model == 'rf':
            self.clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=r_state)
            # specify parameters and distributions to sample from
            param_dist = {
                "max_depth": [2, 4, 8, 16, 32, 64, 128, None],
                "max_features": stats.randint(1, 512),
                "min_samples_split": stats.randint(2, 512),
                "min_samples_leaf": stats.randint(2, 512),
                "criterion": ["gini", "entropy"],
            }
        elif self.args.model == 'bernoulli_nb':
            self.clf = BernoulliNB()
            param_dist = {
                "alpha": stats.uniform(),
                "binarize": stats.uniform(),
                "fit_prior": [True, False],
            }
        else:
            raise ValueError("The model {} is not implemented for random search".format(self.args.model))
        return self._random_search(param_dist, n_iter_search=n_iter_search, r_state=r_state)

    def _random_search(self, param_dist, n_iter_search=20, r_state=1337):
        random_search = RandomizedSearchCV(self.clf,
                                           param_distributions=param_dist,
                                           n_iter=n_iter_search,
                                           cv=10,
                                           scoring="f1_macro",
                                           random_state=r_state,
                                           verbose=2,
                                           n_jobs=-1,
                                           )
        random_search.fit(self.data_x, self.data_y)
        return random_search.best_score_, random_search.best_params_
