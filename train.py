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
    def __init__(self, args, data_x: np.ndarray, data_y: np.ndarray, id2label: list):
        self.args = args
        self.data_x = data_x
        self.data_y = data_y
        self.id2label = id2label
        self.clf = None

    def train(self, small_x=None, small_y=None):
        self._shuffle_data()
        self._create_model()
        self.data_y = MultiLabelBinarizer(classes=[i for i in range(len(self.id2label))]).fit_transform(self.data_y)
        if self.args.train_on_small:
            self._train_on_small_predict_on_large(small_x, small_y)
        if self.args.cross_validate:
            self._cross_validate()
        self.clf.fit(self.data_x, self.data_y)

    def _train_on_small_predict_on_large(self, small_x, small_y):
        print("Train on the small and test on cross-validate test")
        small_y = MultiLabelBinarizer(classes=[i for i in range(len(self.id2label))]).fit_transform(small_y)
        self.clf.fit(small_x, small_y)
        kf = KFold(n_splits=self.args.cv_num)
        acc_list, f1_list = [], []
        for train, test in kf.split(self.data_x, self.data_y):
            X_test = self.data_x[test]
            y_test = self.data_y[test]
            y_predict = self._get_single_label_predict(X_test)
            f1, acc = utils.evaluate_any_type(y_test, y_predict, self.id2label)
            f1_list.append(f1)
            acc_list.append(acc)
        print('The acc score in cross validation is {0}'.format(acc_list))
        print('The f1 score in cross validation is {0}'.format(f1_list))
        print('The average acc score is {0}'.format(np.mean(acc_list)))
        print('The average f1 score is {0}'.format(np.mean(f1_list)))
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
        predict_res = self.clf.predict(data_x)
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
        class_weight = None if self.args.no_class_weight else 'balanced'
        model_name = self.args.model
        print_to_log("The model used here is {0}".format(model_name))
        if model_name == 'sgd':
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
        self.clf = OneVsRestClassifier(clf)

    def _cross_validate(self):
        """
        Don't worry about stratified K-fold, because for cross_validate,
            if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used
        Todo: If you want to get more balanced k-fold split, you can refer to `proba_mass_split` in utils.py
        :return:
        """
        print_to_log('Use {} fold cross validation'.format(self.args.cv_num))
        kf = KFold(n_splits=self.args.cv_num)  # As StratifiedKFold doesn't support multi-label setting
        acc_list, f1_list = [], []
        for train, test in kf.split(self.data_x, self.data_y):
            X_train = self.data_x[train]
            y_train = self.data_y[train]
            X_test = self.data_x[test]
            y_test = self.data_y[test]
            self.clf.fit(X_train, y_train)
            y_predict = self._get_single_label_predict(X_test)
            f1, acc = utils.evaluate_any_type(y_test, y_predict, self.id2label)
            f1_list.append(f1)
            acc_list.append(acc)
        print_to_log('The acc score in cross validation is {0}'.format(acc_list))
        print_to_log('The f1 score in cross validation is {0}'.format(f1_list))
        print_to_log('The average acc score is {0}'.format(np.mean(acc_list)))
        print_to_log('The average f1 score is {0}'.format(np.mean(f1_list)))
        quit()

    def _get_single_label_predict(self, X_test):
        if 'svm' in self.args.model:
            predict_score = self.clf.decision_function(X_test)
        else:
            predict_score = self.clf.predict_proba(X_test)
        return np.argmax(predict_score, axis=-1)

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
