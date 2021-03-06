import os
from typing import List, Dict
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import utils


class Preprocess(object):
    def __init__(self, args, tweetid_list: List[str], tweet_content_list: List[dict], label2id: dict,
                 tweet_id_out_file: str, test=False):
        """
        Use feature_used to control which features are used for sentence-level feature extraction.
            Currently available features:
                ['hand_crafted', 'fasttext-avg', 'fasttext-tfidf', 'glove-avg', 'glove-tfidf', 'cbnu_user_feature',
                'skip-thought', 'bert-avg/CLS-1/4/8', 'fasttext-crawl', 'fasttext-1_2M-balanced-event', 'hashtag']
        :param args:
        :param tweetid_list:
        :param tweet_content_list:
        :param label2id:
        """
        self.args = args
        self.tweetid_list = tweetid_list
        self.tweet_content_list = tweet_content_list
        self.annotated_user_type = None
        self.label2id = label2id
        self.tweet_id_out_file = tweet_id_out_file
        self.test = test
        self.train_tweet = []
        self.train_label = []
        self.tweetid2feature = dict()
        self.feature_len = None
        self.feature_collection = []
        self.feature_used = ['hand_crafted', 'fasttext-avg', 'skip-thought', 'bert-avg-1', 'bert-CLS-1',
                             'glove-tfidf', 'fasttext-crawl']
        # Convert the priority label to score which will be used to train regression model.
        self.priority2score = {'Low': 0.25, 'Medium': 0.5, 'High': 0.75, 'Critical': 1.0, 'Unknown': 0.5}
        utils.print_to_log("The feature used is {}".format(self.feature_used))

    def _collect_feature(self, feature, feat_name):
        self.feature_collection.append(feature)
        utils.print_to_log("The shape of {0}_feature is {1}".format(feat_name, feature.shape))

    def _read_feature_from_file(self, feat_name: str):
        """
        Notice all feature files of 2019 test data should have a postfix, such as "fasttext-crawl-vec-2019.npy"
        It is specifed in the get_embedding.sh in each folder in 'feature_tools'
        :param feat_name:
        :return:
        """
        vecfile_postfix = "-2019" if self.test else ""
        tweetid2vec = utils.get_tweetid2vec(self.tweet_id_out_file, self.args.out_dir, feat_name, vecfile_postfix)
        feature = utils.extract_feature_by_dict(self.tweetid_list, tweetid2vec, feat_name)

        if self.args.use_pca:
            pca_model_filename = 'pca_{}.pkl'.format(feat_name)
            if self.test:
                with open(pca_model_filename, 'rb') as f:
                    pca = pickle.load(f)
                    feature = pca.transform(feature)
            else:
                pca = PCA(n_components=self.args.pca_dim)
                feature = pca.fit_transform(feature)
                with open(pca_model_filename, 'wb') as f:
                    pickle.dump(pca, f)

        if self.args.normalize_feat:
            feature = normalize(feature)

        self._collect_feature(feature, feat_name)

    def extract_features(self):
        """
        Use feature_len to make the late-fusion compatible with early-fusion (first concatenate all features)
        Extract features for all tweetids in self.tweetid_list
        :return:
        """
        hand_crafted_feature, clean_texts = utils.extract_hand_crafted_feature(self.tweet_content_list)
        if "cbnu_user_feature" in self.feature_used:
            # read annotated user type from external file
            annotated_user = os.path.join(self.args.data_dir, 'freq_users.txt')
            self.annotated_user_type = {}
            for line in open(annotated_user):
                id_str, user_type = line.strip().split('\t')
                self.annotated_user_type[id_str] = user_type
            cbnu_user_feature = utils.extract_cbnu_user_feature(self.tweet_content_list,self.annotated_user_type)

        for feat_name in self.feature_used:
            if feat_name == 'hand_crafted':
                self._collect_feature(hand_crafted_feature, feat_name)
            elif feat_name == 'cbnu_user_feature':
                self._collect_feature(cbnu_user_feature, feat_name)
            else:
                self._read_feature_from_file(feat_name)

        # Concatenate all features, and record the length of each feature for future use (such as train model for each)
        whole_feature_matrix = np.concatenate(self.feature_collection, axis=-1)
        if self.args.late_fusion:
            self.feature_len = [it_feature.shape[-1] for it_feature in self.feature_collection]
        else:  # If not use late fusion, we treat the concatenated feature as a whole feature
            self.feature_len = [whole_feature_matrix.shape[-1]]

        assert len(self.tweetid_list) == whole_feature_matrix.shape[0], \
            "The number of tweets is inconsistent, please check if you re-generate features for the new tweetid list"
        for i, tweetid in enumerate(self.tweetid_list):
            self.tweetid2feature[tweetid] = whole_feature_matrix[i]

    def extract_train_data(self, train_file, get_score=False):
        """
        The original design is aiming at extracting the class label. However, as we need to also train the regression
        model on the score, we add a flag 'get_score' which by default compatible with original code.

        :param train_file: File to extract information.
        :param get_score: If False, extract the class; If True, extract the priority score for regression.
        :return:
        """
        if get_score:
            return self._extract_score_from_formalized_file(train_file)
        else:
            return self._extract_data_from_formalized_file_v2(train_file)

    def _extract_score_from_formalized_file(self, filename: str):
        """
        For extracting score, we currently doesn't support event_wise.
        All other things are similar to `_extract_data_from_formalized_file_v2`.

        :param filename: File to extract information.
        :return:
        """
        data_x, data_y = [], []
        count_unk = 0
        with open(filename, 'r', encoding='utf8') as f:
            for idx, line in enumerate(f):
                line = line.strip().split('\t')
                tweetid = line[0]
                priority_label = line[2]
                if priority_label == 'Unknown':
                    count_unk += 1
                score = self.priority2score[line[2]]
                feature = self.tweetid2feature[tweetid]
                data_x.append(feature)
                data_y.append(score)
        utils.print_to_log("There are {} Unknown priority labels.".format(count_unk))
        return np.asarray(data_x), np.asarray(data_y)

    def _extract_data_from_formalized_file_v2(self, filename: str):
        """
        For the new data released in 2019, there is no tweets missed, so we don't need to care about missing tweets
            If you want to deal with missing tweets, please refer to `_extract_data_from_formalized_file_v1`
        Extract data in the form of multi-label, where we treat each "tweet" as a training instance, and the label is
            recorded as a list (such as data_x = [tweetid_1_feature, tweetid_2_feature], data_y = [[0, 2, 5], [1, 5]])
        Notice that if event_wise is True, we store data_x and data_y for each event separately.

        :param filename: File to extract information.
        :return:
        """
        if self.args.event_wise:
            data_x = {event_type: [] for event_type in utils.idx2event_type}
            data_y = {event_type: [] for event_type in utils.idx2event_type}
            event2idx_list = {event_type: [] for event_type in utils.idx2event_type}
        else:
            data_x, data_y = [], []

        with open(filename, 'r', encoding='utf8') as f:
            for idx, line in enumerate(f):
                line = line.strip().split('\t')
                tweetid = line[0]
                event_type = line[3]
                categories = [self.label2id[label] for label in line[1].split(',')]
                feature = self.tweetid2feature[tweetid]
                if self.args.event_wise:
                    data_x[event_type].append(feature)
                    data_y[event_type].append(categories)
                    event2idx_list[event_type].append(idx)
                else:
                    data_x.append(feature)
                    data_y.append(categories)

        if self.args.event_wise:
            for event_type in utils.idx2event_type:
                data_x[event_type] = np.asarray(data_x[event_type])
                data_y[event_type] = np.asarray(data_y[event_type])
            return data_x, data_y, event2idx_list, idx + 1
        else:
            return np.asarray(data_x), np.asarray(data_y)

    def _extract_data_from_formalized_file_v1(self, filename: str):
        """
        This function is deprecated
        :param filename:
        :return:
        """
        count_miss = 0
        count_total = 0

        if self.args.event_wise:
            data_x = {event_type: [] for event_type in utils.idx2event_type}
            data_y = {event_type: [] for event_type in utils.idx2event_type}
            event2idx_list = {event_type: [] for event_type in utils.idx2event_type}
        else:
            data_x, data_y = [], []

        with open(filename, 'r', encoding='utf8') as f:
            for idx, line in enumerate(f):
                line = line.strip().split('\t')
                tweetid = line[0]
                event_type = line[3]
                # The 2018train + 2018test data will not filter out any label
                categories = [self.label2id[label] for label in line[1].split(',')]
                count_total += 1
                if tweetid in self.tweetid2feature:
                    feature = self.tweetid2feature[tweetid]
                    if self.args.event_wise:
                        data_x[event_type].append(feature)
                        data_y[event_type].append(categories)
                        event2idx_list[event_type].append(idx)
                    else:
                        data_x.append(feature)
                        data_y.append(categories)
                else:
                    count_miss += 1

        utils.print_to_log("There are {0}/{1} tweets cannot find for {2}".format(count_miss, count_total, filename))
        if self.args.event_wise:
            for event_type in utils.idx2event_type:
                data_x[event_type] = np.asarray(data_x[event_type])
                data_y[event_type] = np.asarray(data_y[event_type])
            return data_x, data_y, event2idx_list, idx + 1
        else:
            return np.asarray(data_x), np.asarray(data_y)

    def extract_formalized_test_data(self, test_file: str):
        """
        Notice that here the test file should be the formalized file, as the format of training data
            but the label could be some fake labels (because we don't have label for the test data).
        Another thing worth to mention is we don't check if the tweet is in self.tweetid2feature, because
            we assume all tweets provided as the test data should have its content also provided.
        :param test_file:
        :return: If event-wise, we need to return the idx_list for each event,
                    as well as the number of lines in the file
        """
        if self.args.event_wise:
            data_x = {event_type: [] for event_type in utils.idx2event_type}
            event2idx_list = {event_type: [] for event_type in utils.idx2event_type}
        else:
            data_x = []

        with open(test_file, 'r', encoding='utf8') as f:
            for idx, line in enumerate(f):
                line = line.strip().split('\t')
                tweetid = line[0]
                event_type = line[3]
                feature = self.tweetid2feature[tweetid]
                if self.args.event_wise:
                    data_x[event_type].append(feature)
                    event2idx_list[event_type].append(idx)
                else:
                    data_x.append(feature)

        if self.args.event_wise:
            for event_type in utils.idx2event_type:
                data_x[event_type] = np.asarray(data_x[event_type])
            return data_x, event2idx_list, idx + 1
        else:
            return np.asarray(data_x)

    def _extract_data_from_formalized_file_single_label(self, filename: str):
        """
        Note: This function is deprecated, because now we focus on multi-label model, and to make it consistent
            with the official evaluation file, we need our ground truth label in the form of multi-label
        Notice that each tweet may have several labels, and we use each of them to construct a training instance
        :param filename: The filename of formalized file, where each line is "{tweetid}\t{labels}\t{priority}}"
        :return:
        """
        count_miss = 0
        count_total = 0
        data_x, data_y = [], []
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                tweetid = line[0]
                categories = line[1].split(',')
                count_total += 1
                if tweetid in self.tweetid2feature:
                    feature = self.tweetid2feature[tweetid]
                    for tweet_label in categories:
                        if tweet_label not in self.label2id:
                            continue
                        data_x.append(feature)
                        data_y.append(self.label2id[tweet_label])
                else:
                    count_miss += 1

        utils.print_to_log("There are {0}/{1} tweets cannot find for {2}".format(count_miss, count_total, filename))
        data_x, data_y = np.asarray(data_x), np.asarray(data_y, dtype=np.int32)
        print("The shape of data_x is {0}, shape of data_y is {1}".format(data_x.shape, data_y.shape))
        return data_x, data_y

    def extract_test_data_with_labels(self, test_file):
        """
        Note: This function is deprecated, which is used to extract the test data with labels.
        However, currently we direct merge train and test files in cross-validation mode, so we only need to extract
            data and labels from one file (the '2018-all.txt').
        :param test_file:
        :return:
        """
        return self._extract_data_from_formalized_file_v1(test_file)

    def extract_test_data(self, test_file: str):
        """
        Note: This function is deprecated, which is used for the old setting (train on 2018-train and test on 2018-test)
        This function extracts only X data for testing (assume labels are invisible for us)
        It also returns many other auxiliary things which are useful during prediction
        :param test_file:
        :return:
        """
        tweetid_list = []
        miss_tweetid = []
        tweetid2idx = dict()
        tweetid2incident = dict()
        test_x = []

        with open(test_file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                incident_id, tweetid = line[0], line[2]
                tweetid_list.append(tweetid)
                tweetid2incident[tweetid] = incident_id

        tweetid_list = list(set(tweetid_list))  # Remove some duplicate tweets
        for tweetid in tweetid_list:
            if tweetid in self.tweetid2feature:
                tweetid2idx[tweetid] = len(test_x)
                test_x.append(self.tweetid2feature[tweetid])
            else:
                miss_tweetid.append(tweetid)

        utils.print_to_log("There are {0}/{1} tweets cannot find for {2}".format(
            len(miss_tweetid), len(tweetid_list), test_file))
        test_x = np.asarray(test_x)
        utils.print_to_log("The shape of test_x is {0}".format(test_x.shape))

        return test_x, tweetid_list, tweetid2idx, tweetid2incident
