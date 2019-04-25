import random
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

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


if __name__ == '__main__':
    random_seed = 9
    random.seed(random_seed)
    np.random.seed(random_seed)
    test_stratify_split()
