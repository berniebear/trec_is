import random
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier

from utils import stratify_split


def test_stratify_split():
    cv_num = 5
    data = [[0, 1], [1, 3], [2, 3], [0, 2], [0, 1, 2], [0, 2, 3], [1, 3]]
    stratified_data_ids, stratified_data = stratify_split(data, classes=[0, 1, 2, 3], ratios=[1 / cv_num] * cv_num)
    print(stratified_data_ids)
    print(stratified_data)
    print()

    data = MultiLabelBinarizer(classes=[0, 1, 2, 3]).fit_transform(data)
    stratified_data_ids, stratified_data = stratify_split(data, classes=[0, 1, 2, 3],
                                                          ratios=[1 / cv_num] * cv_num, one_hot=True)
    print(stratified_data_ids)
    print(stratified_data)

    # Make it consistent with the kFold API of sklearn
    index_list = []
    for i in range(cv_num):
        test_idx = stratified_data_ids[i]
        train_idx = []
        for ii in range(cv_num):
            if ii == i:
                continue
            train_idx += stratified_data_ids[ii]
        index_list.append((train_idx, test_idx))

    for train, test in index_list:
        print(train)
        print(test)
        print()


def test_multilabel_classify():
    rf = RandomForestClassifier(class_weight=[{0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}])
    # rf = RandomForestClassifier(class_weight={0: 1, 1: 1, 2: 1})
    # rf = OneVsRestClassifier(rf)
    # rf = MultiOutputClassifier(rf)

    x = np.asarray([[0.2, 0.3], [0.4, 0.5]])
    y = np.asarray([[0], [1, 2]])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)

    rf.fit(x, y)
    print(rf.predict_proba(np.asarray([[-10.0, -10.0], [0.7, 0.5]])))
    print(rf.classes_)


if __name__ == '__main__':
    random_seed = 9
    random.seed(random_seed)
    np.random.seed(random_seed)
    test_multilabel_classify()
