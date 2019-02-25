import os
import json
import numpy as np

import utils


class Preprocess():
    def __init__(self, args, tweetid2content: dict, label2id: dict):
        self.args = args
        self.tweetid2content = tweetid2content
        self.label2id = label2id
        self.train_tweet = []
        self.train_label = []

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
        utils.print_to_log("There are {} tweets cannot find in tweetid2content".format(count_miss))

    def content_to_feature(self):
        """
        After extracting the handcrafted features, we also get the clean text,
            which could be used in other feature extractor. And notice that here the text contains uppercase
        :return:
        """
        hand_crafted_feature, clean_texts = utils.extract_hand_crafted_feature(self.train_tweet)
        tfidf_feature = utils.extract_by_tfidf(clean_texts)
        # fasttext_feature = utils.extract_by_fasttext(clean_texts)
        # skip_thought_feature = utils.extract_by_skip_thought(clean_texts)
        # bert_feature = utils.extract_by_bert(clean_texts)
        utils.print_to_log("The shape of hand_crafted_feature is {}".format(hand_crafted_feature.shape))
        utils.print_to_log("The shape of tfidf_feature is {}".format(tfidf_feature.shape))
        data_x = np.concatenate([hand_crafted_feature, tfidf_feature], axis=1)
        data_y = np.asarray(self.train_label, dtype=np.int32)
        utils.print_to_log("The shape of data_x is {0}, data_y is {1}".format(data_x.shape, data_y.shape))
        utils.print_to_log("The number of each label is {}".format({i: sum(data_y == i) for i in range(len(self.label2id))}))
        return data_x, data_y
