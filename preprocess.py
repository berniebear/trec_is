import os
import json
import numpy as np
from sklearn.externals import joblib

import utils


class Preprocess(object):
    def __init__(self, args, tweetid2content: dict, label2id: dict):
        self.args = args
        self.tweetid2content = tweetid2content
        self.label2id = label2id
        self.train_tweet = []
        self.train_label = []
        self.vectorizer = self._get_tfidf_vectorizer()

    def _get_tfidf_vectorizer(self):
        if self.args.sanity_check:
            from sklearn.feature_extraction.text import TfidfVectorizer
            utils.print_to_log("Sanity check mode, use small data for vectorizer")
            vectorizer = TfidfVectorizer()
            vectorizer.fit(['This is a test document', 'This is just used for testing'])
        else:
            vectorizer = joblib.load('../data/2013to2016_tfidf_vectorizer_20190109.pkl')
        return vectorizer

    def remove_all_data(self):
        self.train_tweet = []
        self.train_label = []

    def extract_test_data(self, test_file: str):
        test_content = []
        tweetid_list = []
        miss_tweetid = []
        tweetid2idx = dict()
        tweetid2incident = dict()
        with open(test_file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                incident_id, tweetid = line[0], line[2]
                tweetid_list.append(tweetid)
                tweetid2incident[tweetid] = incident_id
        tweetid_list = list(set(tweetid_list))  # Remove some duplicate tweets
        for tweetid in tweetid_list:
            if tweetid in self.tweetid2content:
                tweetid2idx[tweetid] = len(test_content)
                test_content.append(self.tweetid2content[tweetid])
            else:
                miss_tweetid.append(tweetid)
        utils.print_to_log("There are {0}/{1} tweets cannot find for {2}".format(
            len(miss_tweetid), len(tweetid_list), test_file))
        test_data_x = utils.extract_feature(test_content, self.vectorizer)
        utils.print_to_log("The shape of data_x is {0}".format(test_data_x.shape))
        return test_data_x, tweetid_list, tweetid2idx, tweetid2incident

    def extract_train_data(self, train_file: str):
        """
        Notice that each tweet may have several labels, and we use each of them to construct a training instance
        :param train_file:
        :return:
        """
        count_miss = 0
        with open(train_file, 'r', encoding='utf8') as f:
            train_file_content = json.load(f)
            for event in train_file_content['events']:
                for tweet in event['tweets']:
                    if tweet['postID'] in self.tweetid2content:
                        tweet_content = self.tweetid2content[tweet['postID']]
                        for tweet_label in tweet['categories']:
                            if tweet_label not in self.label2id:
                                continue
                            self.train_tweet.append(tweet_content)
                            self.train_label.append(self.label2id[tweet_label])
                    else:
                        count_miss += 1
        utils.print_to_log("There are {0} tweets cannot find for {1}".format(count_miss, train_file))

    def content_to_feature(self):
        """
        After extracting the handcrafted features, we also get the clean text,
            which could be used in other feature extractor. And notice that here the text contains uppercase
        :return:
        """
        data_x = utils.extract_feature(self.train_tweet, self.vectorizer)
        data_y = np.asarray(self.train_label, dtype=np.int32)
        utils.print_to_log("The shape of data_x is {0}, data_y is {1}".format(data_x.shape, data_y.shape))
        utils.print_to_log("The number of each label is {}".format({i: sum(data_y == i) for i in range(len(self.label2id))}))
        return data_x, data_y
