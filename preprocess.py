import string
import json
from typing import List, Dict
import numpy as np
from sklearn.externals import joblib

import utils


class Preprocess(object):
    def __init__(self, args, tweetid_list: List[str], tweet_content_list: List[dict], label2id: dict):
        self.args = args
        self.tweetid_list = tweetid_list
        self.tweet_content_list = tweet_content_list
        self.label2id = label2id
        self.train_tweet = []
        self.train_label = []
        self.tweetid2feature = dict()

    def extract_features(self):
        """
        Extract features for all tweetids in self.tweetid_list
        :return:
        """
        tfidf_vectorizer = self._get_tfidf_vectorizer()
        fasttext_vectorizer = self._get_fasttext_vectorizer()
        analyzer = tfidf_vectorizer.build_analyzer()

        hand_crafted_feature, clean_texts = utils.extract_hand_crafted_feature(self.tweet_content_list)
        tfidf_feature = utils.extract_by_tfidf(clean_texts, tfidf_vectorizer)
        fasttext_feature = utils.extract_by_fasttext(clean_texts, fasttext_vectorizer, analyzer,
                                                     tfidf_feature, tfidf_vectorizer.get_feature_names(),
                                                     self.args.fasttext_merge)
        del tfidf_feature

        tweetid2skip_vec = utils.get_tweetid2vec(self.args.tweet_id_out_file,
                                                 self.args.skipthought_vec_file, 'skip-thought')
        skipthought_feature = utils.extract_feature_by_dict(self.tweetid_list, tweetid2skip_vec, 'skip-thought')
        tweetid2bertvec = utils.get_tweetid2vec(self.args.tweet_id_out_file, self.args.bert_vec_file, 'bert')
        bert_feature = utils.extract_feature_by_dict(self.tweetid_list, tweetid2bertvec, 'bert')

        utils.print_to_log("The shape of hand_crafted_feature is {}".format(hand_crafted_feature.shape))
        utils.print_to_log("The shape of fasttext_feature is {}".format(fasttext_feature.shape))
        utils.print_to_log("The shape of skip_thought_feature is {}".format(skipthought_feature.shape))
        utils.print_to_log("The shape of bert_feature is {}".format(bert_feature.shape))

        # Concatenate all features
        whole_feature_matrix = np.concatenate([hand_crafted_feature, fasttext_feature,
                                               bert_feature, skipthought_feature], axis=-1)
        assert len(self.tweetid_list) == whole_feature_matrix.shape[0]
        for i, tweetid in enumerate(self.tweetid_list):
            self.tweetid2feature[tweetid] = whole_feature_matrix[i]

    def _get_tfidf_vectorizer(self):
        if self.args.sanity_check:
            from sklearn.feature_extraction.text import TfidfVectorizer
            utils.print_to_log("Sanity check mode, use small data for vectorizer")
            vectorizer = TfidfVectorizer()
            vectorizer.fit(['This is a test document', 'This is just used for testing'])
        else:
            vectorizer = joblib.load('../data/2013to2016_tfidf_vectorizer_20190109.pkl')
        return vectorizer

    def _get_fasttext_vectorizer(self):
        from gensim.models.fasttext import FastText

        if self.args.sanity_check:
            fasttext = FastText(size=10, min_count=1, window=1)
            cleaned_text = ['This is a test document', 'This is just used for testing', string.ascii_lowercase]
            fasttext.build_vocab(cleaned_text)
            fasttext.train(cleaned_text, total_examples=fasttext.corpus_count, epochs=fasttext.epochs)
            return fasttext
        else:
            fasttext = FastText.load('../data/text_sample_2013to2016_gensim_200.model')
            return fasttext

    def extract_train_data(self, train_file: str):
        """
        Notice that each tweet may have several labels, and we use each of them to construct a training instance
        :param train_file:
        :return:
        """
        count_miss = 0
        count_total = 0
        train_x, train_y = [], []
        with open(train_file, 'r', encoding='utf8') as f:
            train_file_content = json.load(f)
            for event in train_file_content['events']:
                for tweet in event['tweets']:
                    count_total += 1
                    tweetid = tweet['postID']
                    if tweetid in self.tweetid2feature:
                        feature = self.tweetid2feature[tweetid]
                        for tweet_label in tweet['categories']:
                            if tweet_label not in self.label2id:
                                continue
                            train_x.append(feature)
                            train_y.append(self.label2id[tweet_label])
                    else:
                        count_miss += 1

        utils.print_to_log("There are {0}/{1} tweets cannot find for {2}".format(count_miss, count_total, train_file))
        train_x, train_y = np.asarray(train_x), np.asarray(train_y, dtype=np.int32)
        print("The shape of train_x is {0}, shape of train_y is {1}".format(train_x.shape, train_y.shape))

        return train_x, train_y

    def extract_test_data(self, test_file: str):
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
