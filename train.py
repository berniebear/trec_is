import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, OneClassSVM, LinearSVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils import shuffle

from utils import print_to_log


class Train(object):
    def __init__(self, args, data_x: np.ndarray, data_y: np.ndarray):
        self.args = args
        self.data_x = data_x
        self.data_y = data_y
        self.clf = None

    def train(self):
        self._shuffle_data()
        self._create_model()
        self._cross_validate()

    def _shuffle_data(self):
        self.data_x, self.data_y = shuffle(self.data_x, self.data_y)

    def _create_model(self):
        class_weight = None if self.args.no_class_weight else 'balanced'
        model_name = self.args.model
        print_to_log("The model used here is {0}".format(model_name))
        if model_name == 'sgd':
            clf = SGDClassifier(max_iter=1000, tol=1e-3, loss='hinge', class_weight=class_weight, shuffle=self.args.shuffle)
        elif model_name == 'svm_linear':
            clf = LinearSVC(class_weight=class_weight, dual=False)
        elif model_name == 'svm_rbf':
            clf = SVC(kernel='rbf', class_weight=class_weight, gamma='auto')
        elif model_name == 'svm_rbf_scale':
            clf = SVC(kernel='rbf', class_weight=class_weight, gamma='scale')
        elif model_name == 'svm_chi2':
            clf = SVC(kernel=chi2_kernel, class_weight=class_weight)
        elif model_name == 'bernoulli_nb':
            clf = BernoulliNB()
        elif model_name == 'rf':
            clf = RandomForestClassifier()
        else:
            raise NotImplementedError
        self.clf = clf

    def _cross_validate(self):
        """
        Don't worry about stratified K-fold, because for cross_validate,
            if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used
        :return:
        """
        print_to_log('Use {} fold cross validation'.format(self.args.cv_num))
        measure_score_name = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
        scores = cross_validate(self.clf, self.data_x, self.data_y, cv=self.args.cv_num, scoring=measure_score_name,
                                return_train_score=True)  # n_jobs=-1
        for it_name in measure_score_name:
            score_list = scores['test_{}'.format(it_name)]
            assert len(score_list) == self.args.cv_num
            print_to_log('The {0} score in cross validation is {1}'.format(it_name, score_list))
            print_to_log('The average {0} score is {1}'.format(it_name, sum(score_list) / len(score_list)))



