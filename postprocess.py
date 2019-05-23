import os
import json
from typing import List, Dict
import numpy as np

import utils

event2incidentid = {'floodChoco2019': 'TRECIS-CTIT-H-Test-022',
                    'fireAndover2019': 'TRECIS-CTIT-H-Test-023',
                    'earthquakeCalifornia2014': 'TRECIS-CTIT-H-Test-024',
                    'earthquakeBohol2013': 'TRECIS-CTIT-H-Test-025',
                    'hurricaneFlorence2018': 'TRECIS-CTIT-H-Test-026',
                    'shootingDallas2017': 'TRECIS-CTIT-H-Test-027',
                    'fireYMM2016': 'TRECIS-CTIT-H-Test-028'}

old2new = {'Other-PastNews': 'Other-ContextualInformation',
           'Other-ContinuingNews': 'Report-News',
           'Other-KnownAlready': 'Report-OriginalEvent',
           'Report-SignificantEventChange': 'Report-NewSubEvent'}


class PostProcess(object):
    """
    Now we already generate the score prediction by different models, and we can do some post processing,
        which aims to generate the final submission file that can be submitted to TREC-IS 2019
    """
    def __init__(self, args, label2id: Dict[str, int], id2label: List[str], class_weight: List[float],
                 majority_label: str, short2long_label: Dict[str, str],
                 formal_train_file: str, formal_2019_test_file: str,
                 raw_tweets_json_folder: str):
        self.args = args
        self.label2id = label2id
        self.id2label = id2label
        self.class_weight = class_weight
        self.majority_label = majority_label
        self.short2long_label = short2long_label
        self.formal_train_file = formal_train_file
        self.formal_2019_test_file = formal_2019_test_file
        self.raw_tweets_json_folder = raw_tweets_json_folder
        self.postfix = '-event' if self.args.event_wise else ''
        self.model_name = '{0}{1}'.format(self.args.model, self.postfix)
        self.submission_folder = self._prepare_submission_folder()
        self.submission_file = os.path.join(self.submission_folder, 'submission_{}'.format(self.model_name))
        self.dev_label_file = os.path.join(self.args.ensemble_dir, 'dev_label.txt')
        self.dev_predict_file = os.path.join(self.args.ensemble_dir, 'dev_predict_{}.txt'.format(self.model_name))
        self.test_predict_file = os.path.join(self.args.ensemble_dir, 'test_predict_{}.txt'.format(self.model_name))
        self.tweetid2incidentid = self._get_tweetid_to_incidentid()
        self.test_predict = self._read_test_predict()
        self.test_tweetid_list = self._read_test_tweetid()
        self.incidentid_list = sorted(list(event2incidentid.values()))
        assert len(self.test_predict) == len(self.test_tweetid_list)

    def _prepare_submission_folder(self):
        temp = self.args.pick_k if self.args.pick_criteria == 'top' else self.args.pick_threshold
        if self.args.pick_criteria == 'autothre':
            temp = None
        submission_folder = 'submit-{0}-{1}'.format(self.args.pick_criteria, 'None' if temp is None else temp)
        submission_folder = os.path.join(self.args.ensemble_dir, submission_folder)
        if not os.path.isdir(submission_folder):
            os.mkdir(submission_folder)
        return submission_folder

    def _read_test_tweetid(self):
        tweetid_list = []
        with open(self.formal_2019_test_file, 'r', encoding='utf8') as f:
            for line in f:
                tweetid_list.append(line.strip().split('\t')[0])
        return tweetid_list

    def _read_test_predict(self):
        predict_res = []
        with open(self.test_predict_file, 'r', encoding='utf8') as f:
            for line in f:
                predicts = list(map(float, line.strip().split()))
                predict_res.append(predicts)
        return predict_res

    def _read_dev_label_predict(self):
        self.dev_label = []
        self.dev_predict = []
        with open(self.dev_label_file, 'r', encoding='utf8') as f:
            for line in f:
                labels = list(map(int, line.strip().split()))
                self.dev_label.append(labels)
        with open(self.dev_predict_file, 'r', encoding='utf8') as f:
            for line in f:
                predicts = list(map(float, line.strip().split()))
                self.dev_predict.append(predicts)

    def _get_weighted_score(self, threshold: float):
        assert len(self.dev_label) == len(self.dev_predict)
        weighted_score = 0.0
        for i in range(len(self.dev_label)):
            labels = set(self.dev_label[i])
            predicts = set([idx for idx, score in enumerate(self.dev_predict[i]) if score >= threshold])
            for i_label, weight in enumerate(self.class_weight):
                if (i_label in labels) == (i_label in predicts):
                    weighted_score += weight
                else:
                    weighted_score -= weight
        return weighted_score

    def _get_tweetid_to_incidentid(self):
        tweetid2incidentid = dict()
        data_folder = self.raw_tweets_json_folder
        filename_list = utils.get_2019_json_file_list(data_folder)
        for filename in filename_list:
            eventid = filename.split('.')[1]
            incidentid = event2incidentid[eventid]
            with open(os.path.join(data_folder, filename), 'r', encoding='utf8') as f:
                for line in f:
                    content = json.loads(line)
                    tweetid = content["allProperties"]["id"]
                    tweetid2incidentid[tweetid] = incidentid
        return tweetid2incidentid

    def _get_score_of_predictions(self, predictions: List[int]):
        return sum([self.class_weight[idx] for idx in predictions]) / len(predictions)

    def _get_predictions_str(self, predictions: List[int]):
        """
        Notice some labels are changed in 2019, so we need to convert them. The details could be found on
            http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/2019Changes.html
        And to get the same format ["AAA","BBB"] as shown in the sample, we need to do some string format operations
        :param predictions:
        :return:
        """
        count_unk, count_modify = 0, 0
        majority_long_label = self.short2long_label[self.majority_label]
        predictions_str = [self.id2label[predict] for predict in predictions]
        predictions_str = [self.short2long_label[predict_str] for predict_str in predictions_str]
        for i, predict_str in enumerate(predictions_str):
            if predict_str == 'Other-Unknown':
                count_unk += 1
                predict_str = majority_long_label
            for old_label, new_label in old2new.items():
                if predict_str == old_label:
                    count_modify += 1
                    predict_str = new_label
                    break
            predictions_str[i] = predict_str
        predictions_str = ['"{}"'.format(predict_str) for predict_str in predictions_str]
        return '[{}]'.format(','.join(predictions_str)), count_unk, count_modify

    def _write_incidentid2content_list_to_file(self, incidentid2content_list):
        count_unk, count_modify = 0, 0
        with open(self.submission_file, 'w', encoding='utf8') as f:
            for incidentid in self.incidentid_list:
                content_list = incidentid2content_list[incidentid]
                content_list = sorted(content_list, key=lambda x: x[1], reverse=True)
                for i, content in enumerate(content_list):
                    predict_str, temp_count_unk, temp_count_modify = self._get_predictions_str(content[2])
                    count_unk += temp_count_unk
                    count_modify += temp_count_modify
                    f.write("{0}\tQ0\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(incidentid, content[0], i+1, content[1],
                                                                        predict_str, self.model_name))
        print("In predictions, there are {0} old labels been converted to new, "
              "and {1} Unknown and we use {2} to replace them".format(count_modify, count_unk,
                                                                      self.short2long_label[self.majority_label]))
        print("The final submission result has been written to {}".format(self.submission_file))

    def _get_threshold_for_each_class(self) -> List[float]:
        assert len(self.dev_label) == len(self.dev_predict)
        threshold_list = []
        for i_class in range(len(self.id2label)):
            best_threshold = 0.1
            best_weighted_sum = float("-inf")
            for threshold in [x / 10 for x in range(1, 10)]:
                weighted_sum = 0.0
                for i in range(len(self.dev_label)):
                    if (i_class in self.dev_label[i]) == (self.dev_predict[i][i_class] >= threshold):
                        weighted_sum += self.class_weight[i_class]
                    else:
                        weighted_sum -= self.class_weight[i_class]
                if weighted_sum > best_weighted_sum:
                    best_weighted_sum = weighted_sum
                    best_threshold = threshold
            threshold_list.append(best_threshold)
        return threshold_list

    def pick_by_autothre(self):
        """
        For the autothre mode, we pick thresholds for each class separately, to make the weighted sum loss minimum.
        :return:
        """
        self._read_dev_label_predict()
        threshold_list = self._get_threshold_for_each_class()
        incidentid2content_list = {incidentid: [] for incidentid in self.incidentid_list}
        for i, predict_scores in enumerate(self.test_predict):
            tweetid = self.test_tweetid_list[i]
            incidentid = self.tweetid2incidentid[tweetid]
            predictions = []
            for idx, score in enumerate(predict_scores):
                if score >= threshold_list[idx]:
                    predictions.append(idx)
            if len(predictions) == 0:
                predictions.append(np.argmax(predict_scores))
            line_contents = [tweetid, self._get_score_of_predictions(predictions), predictions]
            incidentid2content_list[incidentid].append(line_contents)
        print("The threshold for each class is: {}".format(threshold_list))
        self._write_incidentid2content_list_to_file(incidentid2content_list)

    def find_best_threshold(self) -> float:
        self._read_dev_label_predict()
        best_threshold = 0.2
        best_score = 0.0
        candidates = [i / 10 for i in range(2, 10)]
        for threshold in candidates:
            weighted_score = self._get_weighted_score(threshold)
            if weighted_score > best_score:
                best_score = weighted_score
                best_threshold = threshold
        print("After searching threshold in {0}, the best threshold is {1}".format(candidates, best_threshold))
        return best_threshold

    def pick_by_threshold(self, threshold: float):
        """
        Because we already have the confidence score for each class in a "One-vs-Rest" scheme, now we only need to
            determine the threshold and pick those classes above this threshold
        Notice that we want to make sure the submission file is ordered by incident id,
            and inner the incident id it is ranked by score predicted (as the sample shown on website)
        :param threshold:
        :return:
        """
        incidentid2content_list = {incidentid: [] for incidentid in self.incidentid_list}
        for i, predict_scores in enumerate(self.test_predict):
            tweetid = self.test_tweetid_list[i]
            incidentid = self.tweetid2incidentid[tweetid]
            predictions = []
            for idx, score in enumerate(predict_scores):
                if score >= threshold:
                    predictions.append(idx)
            if len(predictions) == 0:
                predictions.append(np.argmax(predict_scores))
            line_contents = [tweetid, self._get_score_of_predictions(predictions), predictions]
            incidentid2content_list[incidentid].append(line_contents)
        print("We pick the result by threshold {}".format(threshold))
        self._write_incidentid2content_list_to_file(incidentid2content_list)

    def pick_top_k(self, k: int):
        """
        Because the threshold may be difficult to determine, we can use the top k classes with highest confidence as
            the output. The k had better smaller than 4.
        :param k:
        :return:
        """
        incidentid2content_list = {incidentid: [] for incidentid in self.incidentid_list}
        for i, predict_scores in enumerate(self.test_predict):
            tweetid = self.test_tweetid_list[i]
            incidentid = self.tweetid2incidentid[tweetid]
            predictions = np.asarray(predict_scores).argsort()[-k:][::-1].tolist()
            line_contents = [tweetid, self._get_score_of_predictions(predictions), predictions]
            incidentid2content_list[incidentid].append(line_contents)
        print("We pick the result by top {}".format(k))
        self._write_incidentid2content_list_to_file(incidentid2content_list)
