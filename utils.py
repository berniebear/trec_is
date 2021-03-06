import os
import re
import logging
import json

from typing import List, Dict
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# For processing tweets
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from postprocess import PostProcess

porter = PorterStemmer()
tweet_tokenizer = TweetTokenizer()
sentiment_analyzer = SentimentIntensityAnalyzer()

logger = logging.getLogger()

idx2event_type = ['boombing', 'earthquake', 'flood', 'typhoon', 'wildfire', 'shooting']
event2type = {'costaRicaEarthquake2012': 'earthquake',
              'fireColorado2012': 'wildfire',
              'floodColorado2013': 'flood',
              'typhoonPablo2012': 'typhoon',
              'laAirportShooting2013': 'shooting',
              'westTexasExplosion2013': 'boombing',
              'guatemalaEarthquake2012': 'earthquake',
              'italyEarthquakes2012': 'earthquake',
              'philipinnesFloods2012': 'flood',
              'albertaFloods2013': 'flood',
              'australiaBushfire2013': 'wildfire',
              'bostonBombings2013': 'boombing',
              'manilaFloods2013': 'flood',
              'queenslandFloods2013': 'flood',
              'typhoonYolanda2013': 'typhoon',
              'joplinTornado2011': 'typhoon',
              'chileEarthquake2014': 'earthquake',
              'typhoonHagupit2014': 'typhoon',
              'nepalEarthquake2015': 'earthquake',
              'flSchoolShooting2018': 'shooting',
              'parisAttacks2015': 'boombing',  # The following seven lines are for 2019-A.
              'floodChoco2019': 'flood',
              'fireAndover2019': 'wildfire',
              'earthquakeCalifornia2014': 'earthquake',
              'earthquakeBohol2013': 'earthquake',
              'hurricaneFlorence2018': 'typhoon',
              'shootingDallas2017': 'shooting',
              'fireYMM2016': 'wildfire',  # The following six lines are for 2019-B.
              'albertaWildfires2019': 'wildfire',
              'cycloneKenneth2019': 'typhoon',
              'philippinesEarthquake2019': 'earthquake',
              'coloradoStemShooting2019': 'shooting',
              'southAfricaFloods2019': 'flood',
              'sandiegoSynagogueShooting2019': 'shooting'}
assert len(set(event2type.values())) == 6

# Label adjustment according to http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/2019Changes.html
short_old2new_label = {'PastNews': 'ContextualInformation',
                       'ContinuingNews': 'News',
                       'KnownAlready': 'OriginalEvent',
                       'SignificantEventChange': 'NewSubEvent'}
# Priority weights according to
# http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/TREC%202019-A%20Incident%20Streams%20Track.pdf
priority2score = {'Irrelevant': 0.0, 'Low': 0.25, 'Medium': 0.5, 'High': 0.75, 'Critical': 1.0}
informative_categories = ['GoodsServices', 'SearchAndRescue', 'MovePeople',
                          'EmergingThreats', 'NewSubEvent', 'ServiceAvailable']


class GloveVectorizer(object):
    def __init__(self, word2vecs: Dict[str, List[float]], vector_size: int):
        self.wv = word2vecs
        self.vector_size = vector_size


def set_logging(args):
    # create logger, if you call it in other script, it will retrieve this logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(args.log_dir, '{}.log'.format(args.log_name)))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # logger.debug('debug message')
    # logger.info('info message')
    # logger.warning('warning message')
    # logger.error('error message')
    # logger.critical('critical message')
    return logger


def prepare_folders(args):
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.isdir(args.test_dir):
        os.mkdir(args.test_dir)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.isdir(args.ensemble_dir):
        os.mkdir(args.ensemble_dir)


def prepare_submission_folder(args):
    temp = args.pick_k if args.pick_criteria == 'top' else args.pick_threshold
    if args.pick_criteria == 'autothre':
        temp = None
    submission_folder = 'submit-{0}-{1}'.format(args.pick_criteria, 'None' if temp is None else temp)
    submission_folder = os.path.join(args.ensemble_dir, submission_folder)
    if not os.path.isdir(submission_folder):
        os.mkdir(submission_folder)
    return submission_folder


def check_args_conflict(args):
    """
    Because some arguments cannot be set to True or False together
    :param args:
    :return:
    """
    assert args.late_fusion is False, "To make the code easier to modify, we don't support late-fusion any more"
    assert args.train_on_small is False, "This function is deprecated"
    if args.event_wise:
        assert args.train_regression is False, "Currently we don't support event-wise regression."
        assert args.cross_validate is False, "Currently for event-wise, we don't support cross_validate."
    if args.predict_mode:
        assert args.late_fusion is False, "Currently our predict model only supports early-fusion"
        assert args.search_best_parameters is False
        assert args.train_on_small is False
    if args.search_best_parameters:
        assert args.event_wise is False, "Current model doesn't support search best parameter for event-wise model." \
                                         "We recommend to search parameter for general model and " \
                                         "directly apply the event-wise setting with the best parameter."
        assert args.search_by_sklearn_api is False, "Please use our own method instead of the sklearn API."
    if args.class_weight_scheme == 'customize':
        assert args.model == 'rf', "We only support rf for customized weight, because all other models need to be" \
                                   "wrapped with OneVsRestClassifier, which is not compatible with customized" \
                                   "class_weight in current sklearn version."

    if args.merge_priority_score != 'simple':
        assert args.merge_priority_score == 'advanced'
        assert args.train_regression, "The advanced method must use both regression and prediction."
        assert 0.0 <= args.advanced_predict_weight <= 1.0
    if args.sanity_check:
        args.n_jobs = 1


def get_class_weight(args, label2id: Dict[str, int], id2label: List[str], formal_train_file: str) -> List[float]:
    """
    As the host says they will pay more attention to the "important" classes, we want to calculate the weight
        according to training file.
    The original expression in official webiste is:
        We mostly care about getting actionable information to the response officer,
        and less about other categories like sentiment or general news reporting

    :return: A dictionary to map from class index to class weight (name of the class could be got by id2label)
    """
    class_weight = [1.0 / len(label2id)] * len(label2id)
    class_sum_score = {i: 0.0 for i in range(len(label2id))}
    class_count = {i: 0 for i in range(len(label2id))}
    priority_unk_count = 0
    with open(formal_train_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            class_labels, priority = line[1].split(','), line[2]
            if priority == 'Unknown':
                priority_unk_count += 1
                continue
            score = priority2score[priority]
            for label in class_labels:
                idx = label2id[label]
                class_sum_score[idx] += score
                class_count[idx] += 1
    for i in range(len(class_weight)):
        class_weight[i] = class_sum_score[i] / class_count[i]
    # Some informative classes are more important
    for category in informative_categories:
        # There is no requirements that the element in class_weight need to be smaller than 1.0.
        class_weight[label2id[category]] = class_weight[label2id[category]] + args.additional_weight
    if args.class_weight_scheme == 'customize':
        print("Use customized weight, which gives more weights to the actionable class.")
        print("There are {0} lines have 'Unknown' as priority, just ignored them".format(priority_unk_count))
        print("After calculating class weight, the new weight is:")
        for i_label in np.argsort(class_weight)[::-1]:
            print("{0}: {1}".format(id2label[i_label], class_weight[i_label]))
    return class_weight


def save_variable_to_file(variables_dict: Dict, target_filename):
    with open(target_filename, 'wb') as f:
        pickle.dump(variables_dict, f)
    print("Those variables have been saved to {1}: {0}".format(variables_dict.keys(), target_filename))


def write_predict_and_label(args, formal_train_file: str, label2id: Dict[str, int], data_predict_collect: np.ndarray):
    """
    For each call of this function, we will write the dev labels (multiple labels per line)
        and dev predict (the probability score per line) to the file.
    Those files can be used during the ensemble, because the dev features and dev labels could be used to train it.
    :return:
    """
    dev_label_file = args.dev_label_file
    dev_predict_file = args.dev_predict_file

    # Write the dev label file according to formal_train_file
    fout = open(dev_label_file, 'w', encoding='utf8')
    with open(formal_train_file, 'r', encoding='utf8') as f:
        for line in f:
            label_list = line.strip().split('\t')[1].split(',')
            label_id_list = [label2id[label] for label in label_list]
            fout.write('{}\n'.format(' '.join(list(map(str, label_id_list)))))
    fout.close()

    with open(dev_predict_file, 'w', encoding='utf8') as f:
        for row in data_predict_collect:
            f.write('{}\n'.format(' '.join(list(map(str, row)))))

    print("The dev labels has been written to {0} and predict has been written to {1}".format(
        dev_label_file, dev_predict_file))


def write_predict_score_to_file(scores, file):
    """
    Note that all writing logic in this program is only writing the value, without corresponding tweetid, because
    the order of the tweetid is fixed, which can be read from the file (like `formal_2019_test_file`).
    So, a signicant thing is to avoid any shuffle in the whole process.
    """
    with open(file, 'w', encoding='utf8') as f:
        for score in scores:
            f.write('{}\n'.format(score))
    print("The predict priority score for test data has been written to {}".format(file))


def write_predict_res_to_file(args, test_predict_collect):
    """
    Write the predict scores for each test data (each row represent a data) to the file.
    :param args: Config arguments that contains the `test_predict_file` which specifies the output file.
    :param test_predict_collect: An ndarray with size [num_instance, num_class].
    :return:
    """
    with open(args.test_predict_file, 'w', encoding="utf8") as f:
        for row in test_predict_collect:
            f.write('{}\n'.format(' '.join(list(map(str, row)))))
    print("The test predict has been written to {0}".format(args.test_predict_file))


def get_predict_file_list(ensemble_dir, prefix):
    """
    Notice we sorted the file to make sure the order is consistent
    :param ensemble_dir:
    :param prefix: such as 'dev_predict_'
    :return:
    """
    predict_file_list = []
    for filename in os.listdir(ensemble_dir):
        if filename[:len(prefix)] == prefix:
            predict_file_list.append(os.path.join(ensemble_dir, filename))
    predict_file_list = sorted(predict_file_list)
    print("There are {0} files for {1}".format(len(predict_file_list), prefix))
    return predict_file_list


def get_ensemble_feature(file_list: List[str]) -> np.ndarray:
    feature_collect = []
    for filepath in file_list:
        feature = []
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f:
                feature.append(list(map(float, line.strip().split())))
        feature_collect.append(np.asarray(feature))
    return np.concatenate(feature_collect, axis=-1)


def get_ensemble_label(label_file: str) -> List[List[int]]:
    label = []
    with open(label_file, 'r', encoding='utf8') as f:
        for line in f:
            label.append(list(map(int, line.strip().split())))
    return label


def ensemble_cross_validate(data_x: np.ndarray, data_y: List[List[int]], id2label: List[str],
                            mlb: MultiLabelBinarizer, ensemble: str):
    data_y = mlb.transform(data_y)
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_values = {metric_name: [] for metric_name in metric_names}

    index_list = get_k_fold_index_list(data_y, id2label, cv_num=5)
    for train_idx_list, test_idx_list in index_list:
        X_train = data_x[train_idx_list]
        y_train = data_y[train_idx_list]
        X_test = data_x[test_idx_list]
        y_test = data_y[test_idx_list]
        predict = ensemble_train_and_predict(X_train, y_train, X_test, id2label, ensemble)
        metric_results = evaluate_any_type(y_test, predict, id2label)
        for metric_name in metric_names:
            metric_values[metric_name].append([metric_results[metric_name], len(y_test)])
    metric_weighted_avg = get_weighted_avg(metric_values, metric_names)
    print_to_log('For ensemble, the model used is {}'.format(ensemble))
    for metric_name in metric_names:
        print_to_log('For ensemble, {0} score in cv is {1}'.format(metric_name, metric_values[metric_name]))
        print_to_log('For ensemblem average {0} score is {1}'.format(metric_name, metric_weighted_avg[metric_name]))


def ensemble_train_and_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                               id2label: List[str], ensemble: str):
    """
    # Todo: Another option is to store the multi-label predicted by each base model, and then do the voting
    Currently, for the voting method, we add all the predict_prob by the model
    :param train_x:
    :param train_y: Notice here train_y need to be the one-hot representation produced by mlb.transform()
    :param test_x:
    :param id2label:
    :param ensemble:
    :return:
    """
    if ensemble == 'voting':
        category_num = len(id2label)
        split_num = test_x.shape[1] / category_num
        split_probas = np.split(test_x, split_num, axis=-1)
        merged_proba = np.sum(np.asarray(split_probas), axis=0)
        predict = np.argmax(merged_proba, axis=-1)
    else:
        if ensemble == 'svm_linear':
            clf = SVC(kernel='linear', probability=True)
        elif ensemble == 'svm_rbf':
            clf = SVC(probability=True)
        elif ensemble == 'logistic_reg':
            clf = LogisticRegression(class_weight='balanced')
        else:
            raise ValueError("Invalid value {} for --ensemble".format(ensemble))
        clf = OneVsRestClassifier(clf, n_jobs=-1)
        clf.fit(train_x, train_y)
        predict_proba = clf.predict_proba(test_x)
        predict = np.argmax(predict_proba, axis=-1)
    return predict


def get_weighted_avg(metric_values, metric_names: List[str]):
    metric_accumulate = {metric_name: 0.0 for metric_name in metric_names}
    count = {metric_name: 0 for metric_name in metric_names}
    for metric_name in metric_names:
        for value, length in metric_values[metric_name]:
            metric_accumulate[metric_name] += value * length
            count[metric_name] += length
    for metric_name in metric_names:
        metric_accumulate[metric_name] /= count[metric_name]
    return metric_accumulate


def get_k_fold_index_list(data_y, id2label: List[str], cv_num: int):
    """
    Generate a index list for the k-fold cross-validation, in the same format as the index returned by sklearn KFold.
    Use the stratified sampling for multi-label setting.
    Notice here the data_y should be the label after being binarized (the one-hot label)
    :param data_y:
    :param id2label:
    :param cv_num:
    :return: The returned value could be used as
                index_list = get_k_fold_index_list()
                for train_idx_list, test_idx_list in index_list:
                    X_train = data_x[train_idx_list]
                    y_train = data_y[train_idx_list]
                    X_test = data_x[test_idx_list]
                    y_test = data_y[test_idx_list]
    """
    stratified_data_ids, _ = stratify_split(data_y, list(range(len(id2label))), [1 / cv_num] * cv_num, one_hot=True)
    index_list = []
    for i in range(cv_num):
        test_idx = stratified_data_ids[i]
        train_idx = []
        for ii in range(cv_num):
            if ii == i:
                continue
            train_idx += stratified_data_ids[ii]
        index_list.append((train_idx, test_idx))
    return index_list


def anytype_f1_scorer(y_true, y_pred, id2label):
    """
    Notice that here y_pred is probability for each class
    :param y_true:
    :param y_pred:
    :param id2label:
    :return:
    """
    y_pred = np.argmax(y_pred, axis=-1)
    score = evaluate_any_type(y_true, y_pred, id2label)['f1']
    return score


def get_final_metrics(metrics_collect, metrics_names: List[str]):
    """
    Get the metrics for each event type, as well as the number of data for each event type, calculate the weighted avg
    :param metrics_collect: List[(Dict[str, float], int)]
    :param metrics_names:
    :return:
    """
    accumulate_res = {metrics_name: 0.0 for metrics_name in metrics_names}
    count = {metrics_name: 0 for metrics_name in metrics_names}
    for metrics, data_num in metrics_collect:
        for metric_name, val in metrics.items():
            accumulate_res[metric_name] += val * data_num
            count[metric_name] += data_num
    for metrics_name in metrics_names:
        accumulate_res[metrics_name] /= count[metrics_name]
    logger.info("The final evaluation metrics val for event-wise model is {}".format(accumulate_res))


def formalize_files(origin_file_list: List[str], formalized_out_file: str, args):
    fout = open(formalized_out_file, 'w', encoding='utf8')
    for test_file in origin_file_list:
        formalize_helper(test_file, fout, args.sanity_check)
    fout.close()


def get_2019_json_file_list(data_folder: str, prefix: str):
    filename_list = []
    for filename in os.listdir(data_folder):
        if filename.startswith(prefix) and filename.endswith(".json"):
            filename_list.append(filename)
    filename_list = sorted(filename_list)
    return filename_list


def formalize_test_file(data_folder: str, formalized_file: str, prefix: str):
    """ Read test json files, and the test files are specified by `prefix`, such as 'trecis2019-B'.

    As the 2019-A test data downloaded by official jar are some separate files, so we need to collect them together
    Notice that we don't know the label and priority of the test data, we use UNK to replace it.
    The 2019-A test data has already been ranked by time, so we just need to read line by line
    """
    filename_list = get_2019_json_file_list(data_folder, prefix)
    fout = open(formalized_file, 'w', encoding='utf8')
    for filename in filename_list:
        eventid = filename.split('.')[1]
        event_type = event2type[eventid]
        with open(os.path.join(data_folder, filename), 'r', encoding='utf8') as f:
            for line in f:
                content = json.loads(line)
                tweetid = content["allProperties"]["id"]
                fout.write('{0}\tUNK\tUNK\t{1}\n'.format(tweetid, event_type))
    fout.close()


def formalize_helper(filename, fout, sanity_check=False):
    """
    Note the 2019 data file provided by TREC uses `latin-1` instead of `utf8` encoding.
    Another thing is that for old data the label set is old (ITR-H.types.v2.json), but as we use the new label set now
        (ITR-H.types.v4.json), we need to convert the old label to new label in advance.
    The "Unknown" is deleted after v2, so we also ignore this label in our data.
    """
    is_2019_data = '2019' in filename
    unk_count = 0
    line_write_count = 0
    with open(filename, 'r', encoding='latin-1' if is_2019_data else 'utf8') as f:
        train_file_content = json.load(f)
        for event in train_file_content['events']:
            eventid = event['eventid']
            # To handle data split of the event name which is not exactly the same as in offcial doc.
            # For example, the `nepalEarthquake2015` may appear as `nepalEarthquake2015S`.
            if eventid not in event2type:
                eventid = eventid[:-1]
            if eventid not in event2type:
                eventid = eventid[:-1]
            for tweet in event['tweets']:
                tweetid = tweet['postID']
                label_list = tweet['categories']
                if not is_2019_data:
                    label_list = [short_old2new_label[label] if label in short_old2new_label else label
                                  for label in label_list]
                    label_list = [label for label in label_list if label != 'Unknown']
                    if len(label_list) == 0:
                        unk_count += 1
                        break
                priority = tweet['priority']
                fout.write('{0}\t{1}\t{2}\t{3}\n'.format(tweetid, ','.join(label_list), priority, event2type[eventid]))
                line_write_count += 1
                if sanity_check:
                    # To make sure all labels has at least cv_num data points (or the cross validation will fail).
                    for _ in range(5):
                        fout.write(
                            '{0}\t{1}\t{2}\t{3}\n'.format(tweetid, ','.join(label_list), priority, event2type[eventid]))
                    if line_write_count >= 50:
                        return
    print("There are {0} lines discarded in {1} because it only has 'Unknown' lebel".format(unk_count, filename))


def merge_files(filenames: List[str], outfile: str):
    with open(outfile, 'w', encoding='utf8') as fout:
        for fname in filenames:
            with open(fname) as infile:
                fout.write(infile.read())


def write_list_to_file(target_list: list, outfile):
    with open(outfile, 'w', encoding='utf8') as f:
        for line in target_list:
            f.write(line + '\n')


def write_tweet_and_ids(tweetid_list: List[str], tweet_content_list: List[dict],
                        tweet_text_out_file: str, tweet_id_out_file: str):
    """
    Write all tweetids as well as contents to files, which is prepared for external feature extraction (such as BERT)
    :param tweetid_list:
    :param tweet_content_list
    :param tweet_text_out_file:
    :param tweet_id_out_file:
    :return:
    """
    clean_texts = []
    for i, content in enumerate(tweet_content_list):
        entities = content['entities']
        clean_text = get_clean_tweet(content['full_text'], entities)
        clean_texts.append(clean_text)
    write_list_to_file(clean_texts, tweet_text_out_file)
    write_list_to_file(tweetid_list, tweet_id_out_file)
    print("The tweet text and ids has been written to {0} and {1}".format(tweet_text_out_file, tweet_id_out_file))


def get_label2id(label_file: str, train_file: str, threshold: int):
    """
    Because the original labels are in the form of text, such as "MultimediaShare" and so on.
    We want to convert those textual labels to digital labels.
    :param label_file: All types of labels provided by TREC, including explanation of each type of label
    :param train_file: Formalized train file, where each line is "{tweetid}\t{labels}\t{Priority}\t{EventType}\n"
    :param threshold: If the appearance number of a label is smaller than the threshold, it will be filtered out.
    :return:
        label2id: the dict to convert labels to id
        majority_label: the majority label, used when the tweet contents cannot be accessed
        short2long_label: for submission we need to predict the long label (MultimediaShare -> Report-MultimediaShare)
    """
    # Parsing the file of label type explanation, get the idx for each type of label
    id2label = []
    short2long_label = dict()
    with open(label_file, 'r', encoding='utf8') as f:
        content = json.load(f)
    for it_type in content['informationTypes']:
        long_label = it_type['id']
        label = long_label.split('-')[1]
        short2long_label[label] = long_label
        if label in id2label:
            raise ValueError("The label {0} duplicate in {1}".format(label, label_file))
        id2label.append(label)
    print_to_log("All labels in {0} is: {1}".format(label_file, id2label))

    # Count label frequency
    label_count = {label: 0 for label in id2label}
    with open(train_file, 'r', encoding='utf8') as f:
        for line in f:
            labels = line.strip().split('\t')[1].split(',')
            for label in labels:
                label_count[label] += 1

    # Get the majority label
    majority_label = id2label[0]
    max_count = label_count[majority_label]
    for label in id2label:
        if label_count[label] > max_count:
            majority_label = label
            max_count = label_count[label]

    # Remove those rare labels
    removed_labels = []
    label2id = dict()
    for label in id2label:
        if label_count[label] < threshold:
            removed_labels.append(label)
        else:
            label2id[label] = len(label2id)
    print_to_log("With threshold {0}, those labels are filtered out: {1}".format(threshold, removed_labels))
    if len(removed_labels) > 0:
        # For using v4 label set and training on 2018 data, the 'Location' will never appear.
        assert len(removed_labels) == 1 and removed_labels[0] == 'Location', \
            "In our current setting, there should be no label removed except for 'Location'."

    return label2id, majority_label, short2long_label


def get_id2label(label2id: dict):
    id2label = [''] * len(label2id)
    for label, idx in label2id.items():
        id2label[idx] = label
    return id2label


def evaluate_2019B(y_test: np.ndarray, predict_score: np.ndarray, high_prior_labels: set, args) -> Dict[str, float]:
    """
    Use the evaluation script for 2019B as a reference to get those metrics.

    We calculate the score for each category, and then average them (with some constraints such as the high prior).
    For a category c, for each tweet if it contains c in ground truth, we will label it as 1, and similar for the
        prediction, and then we calculate the precirsion/recall/f1 for c.

    :param y_test: A binary 2-D matrix where [i,j] entry represents if ith instance has label j.
    :param predict_score: A matrix with size [data_num, class_num] and each entry is a score.
    :param high_prior_labels: The set of the label which belongs to the informative type.
    :return:
    """
    class_num = predict_score.shape[1]
    precision, recall, f1, accuracy = 0.0, 0.0, 0.0, 0.0
    high_prior_precision, high_prior_recall, high_prior_f1, high_prior_accuracy = 0.0, 0.0, 0.0, 0.0
    for class_idx in range(class_num):
        ground_truth = y_test[:, class_idx]
        if args.pick_criteria == 'top':
            predict_label = [PostProcess._pick_top_k(it_predict_score, args.pick_k) for it_predict_score in predict_score]
        else:
            raise NotImplementedError("Currently other pick criteria is not supported for cross-validation")
        predict_label = np.asarray([1 if class_idx in k_labels else 0 for k_labels in predict_label])
        current_precision = precision_score(ground_truth, predict_label, average='binary')
        current_recall = recall_score(ground_truth, predict_label, average='binary')
        current_f1 = f1_score(ground_truth, predict_label, average='binary')
        current_accuracy = accuracy_score(ground_truth, predict_label)

        precision += current_precision
        recall += current_recall
        f1 += current_f1
        accuracy += current_accuracy

        if class_idx in high_prior_labels:
            high_prior_precision += current_precision
            high_prior_recall += current_recall
            high_prior_f1 += current_f1
            high_prior_accuracy += current_accuracy

    return {'precision': precision / class_num, 'recall': recall / class_num, 'f1': f1 / class_num,
            'accuracy': accuracy / class_num, 'high_prior_precision': high_prior_precision / len(high_prior_labels),
            'high_prior_recall': high_prior_recall / len(high_prior_labels),
            'high_prior_f1': high_prior_f1 / len(high_prior_labels),
            'high_prior_accuracy': high_prior_accuracy / len(high_prior_labels)}


def evaluate_weighted_sum(y_test: np.ndarray, predict_score: np.ndarray, class_weight: List[float]) -> Dict[str, float]:
    """
    Because different classes have different weights, here we evaluate the predict score according to different weights
    We use the cross-entropy which is CE = ylogp + (1-y)log(1-p), and this metric is the larger, the better
    :param y_test:
    :param predict_score:
    :param class_weight:
    :return:
    """
    data_num = y_test.shape[0]
    Yis1 = y_test == 1
    epsilon = 1e-15
    tile_class_weight = np.tile(class_weight, [data_num, 1])
    predict_score += epsilon
    weighted_ce = np.sum(np.log(predict_score[Yis1]) * tile_class_weight[Yis1]) + np.sum(np.log(1-predict_score[~Yis1]) * tile_class_weight[~Yis1])
    weighted_ce = np.squeeze(weighted_ce) / data_num
    return {'weighted_ce': weighted_ce}


def evaluate_any_type(label: np.ndarray, predict: np.ndarray, id2label: List[str]) -> Dict[str, float]:
    """
    "Any type" means that when the predict label is equal to any of the labels in ground truth, we view it as correct
    Use the "EVALUATON 8: Information Type Categorization (Any-type)" in evaluate.py as reference
    :param label: A binary 2-D matrix where [i,j] entry represents if ith instance has label j
    :param predict: A 1-D ndarray where each entry is a prediction
    :return: f1 and accuracy
    """
    truePositive = 0  # system predicted any of the categories selected by the human assessor
    trueNegative = 0  # system and human assessor both selected either Other-Irrelevant or Other-Unknown
    falsePositive = 0  # system failed to predict any of the categories selected by the human assessor
    falseNegative = 0  # human assessor selected either Other-Irrelevant or Other-Unknown but the system prediced something different

    for i, predict_it in enumerate(predict):
        categoryMatchFound = False
        isNegativeExample = False
        if label[i][predict_it] == 1:
            categoryMatchFound = True
        if id2label[predict_it] == 'Irrelevant' or id2label[predict_it] == 'Unknown':
            isNegativeExample = True

        if categoryMatchFound & isNegativeExample:
            trueNegative = trueNegative + 1
        if categoryMatchFound & (not isNegativeExample):
            truePositive = truePositive + 1
        if (not categoryMatchFound) & isNegativeExample:
            falseNegative = falseNegative + 1
        if (not categoryMatchFound) & (not isNegativeExample):
            falsePositive = falsePositive + 1

    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    f1 = 2 * ((precision * recall) / (precision + recall))
    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def proba_mass_split(y, folds=7):
    # np.random.seed(1)
    # y = np.random.randint(0, 2, (5000, 5))
    # y = y[np.where(y.sum(axis=1) != 0)[0]]

    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)
    print("Fold distributions are")
    print(fold_dist)
    return index_list


def stratify_split(data, classes, ratios, one_hot=False):
    """Stratifying procedure. Reference: https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/

    data is a list of lists: a list of labels, for each sample.
        Each sample's labels should be ints, if they are one-hot encoded, use one_hot=True

    classes is the list of classes each label can take

    ratios is a list, summing to 1, of how the dataset should be split

    """
    # one-hot decoding
    if one_hot:
        temp = [[] for _ in range(len(data))]
        indexes, values = np.where(np.array(data).astype(int) == 1)
        for k, v in zip(indexes, values):
            temp[k].append(v)
        data = temp

    # Organize data per label: for each label l, per_label_data[l] contains the list of samples
    # in data which have this label
    per_label_data = {c: set() for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].add(i)

    # number of samples
    size = len(data)

    # In order not to compute lengths each time, they are tracked here.
    subset_sizes = [r * size for r in ratios]
    target_subset_sizes = list(subset_sizes)
    per_label_subset_sizes = {
        c: [r * len(per_label_data[c]) for r in ratios]
        for c in classes
    }

    # For each subset we want, the set of sample-ids which should end up in it
    stratified_data_ids = [set() for _ in range(len(ratios))]

    # For each sample in the data set
    while size > 0:
        # Compute |Di|
        lengths = {
            l: len(label_data)
            for l, label_data in per_label_data.items()
        }
        try:
            # Find label of smallest |Di|
            label = min(
                {k: v for k, v in lengths.items() if v > 0}, key=lengths.get
            )
        except ValueError:
            # If the dictionary in `min` is empty we get a Value Error.
            # This can happen if there are unlabeled samples.
            # In this case, `size` would be > 0 but only samples without label would remain.
            # "No label" could be a class in itself: it's up to you to format your data accordingly.
            break
        current_length = lengths[label]

        # For each sample with label `label`
        while per_label_data[label]:
            # Select such a sample
            current_id = per_label_data[label].pop()

            subset_sizes_for_label = per_label_subset_sizes[label]
            # Find argmax clj i.e. subset in greatest need of the current label
            largest_subsets = np.argwhere(
                subset_sizes_for_label == np.amax(subset_sizes_for_label)
            ).flatten()

            if len(largest_subsets) == 1:
                subset = largest_subsets[0]
            # If there is more than one such subset, find the one in greatest need
            # of any label
            else:
                largest_subsets = np.argwhere(
                    subset_sizes == np.amax(subset_sizes)
                ).flatten()
                if len(largest_subsets) == 1:
                    subset = largest_subsets[0]
                else:
                    # If there is more than one such subset, choose at random
                    subset = np.random.choice(largest_subsets)

            # Store the sample's id in the selected subset
            stratified_data_ids[subset].add(current_id)

            # There is one fewer sample to distribute
            size -= 1
            # The selected subset needs one fewer sample
            subset_sizes[subset] -= 1

            # In the selected subset, there is one more example for each label
            # the current sample has
            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1

            # Remove the sample from the dataset, meaning from all per_label dataset created
            for l, label_data in per_label_data.items():
                if current_id in label_data:
                    label_data.remove(current_id)

    # Create the stratified dataset as a list of subsets, each containing the orginal labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    stratified_data = [
        [data[i] for i in strat] for strat in stratified_data_ids
    ]

    # Return both the stratified indexes, to be used to sample the `features` associated with your labels
    # And the stratified labels dataset
    return stratified_data_ids, stratified_data


def get_tweetid_content(tweet_file_list: List[str]) -> (List[str], List[dict]):
    """
    Read the train and test tweets file to form a large set of tweetid along with its content
    :param tweet_file_list:
    :return:
    """
    tweetid_list = []
    tweet_content_list = []
    for filename in tweet_file_list:
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                content = json.loads(line.strip())
                tweetid = content['id_str']
                tweetid_list.append(tweetid)
                tweet_content_list.append(content)
    return tweetid_list, tweet_content_list


def extract_feature_by_dict(tweetid_list: List[str], tweetid2vec: Dict[str, List[float]], feat_name: str) -> np.ndarray:
    """
    :param tweetid_list:
    :param tweetid2vec:
    :param feat_name:
    :return:
    """
    res = []
    tweetid2bertvec_iter = iter(tweetid2vec)
    bert_dim = len(tweetid2vec[next(tweetid2bertvec_iter)])
    miss_num = 0
    for tweetid in tweetid_list:
        if tweetid in tweetid2vec:
            res.append(tweetid2vec[tweetid])
        else:
            res.append([0.0] * bert_dim)
            miss_num += 1
    print_to_log("There are {0}/{1} missed by {2} features".format(miss_num, len(tweetid_list), feat_name))
    return np.asarray(res)


def get_tweetid2vec(tweetid_file: str, vec_dir: str, feat_name: str, vecfile_postfix) -> Dict[str, List[float]]:
    """
    The file of tweets content is tweet_text_out_file
    The file of tweets id is tweetid_file
    Notice the order of the tweet id and tweet content is consistent for those two files
    :param tweetid_file:
    :param vec_dir: The folder contains corresponding feature vector (such as bert or skip-thought)
    :param feat_name:
    :param vecfile_postfix: to distinguish the 2019 vector file with the 2018 original vector file
    :return:
    """
    if feat_name.startswith('bert'):
        vec_file = os.path.join(vec_dir, 'bert-vec{}.json'.format(vecfile_postfix))
        _, bert_type, layer = feat_name.split('-')  # bert-CLS-1 means using -1 layer of CLS feature
        return get_tweetid2bertvec(tweetid_file, vec_file, bert_type, layer)
    else:
        vec_file = os.path.join(vec_dir, '{0}-vec{1}.npy'.format(feat_name, vecfile_postfix))
        return get_tweetid2vec_by_npy(tweetid_file, vec_file)


def get_tweetid2bertvec(tweetid_file: str, bert_vec_file: str, bert_type: str, layer: str) -> Dict[str, List[float]]:
    tweetid2bertvec = dict()
    tweetid_list = []
    feat_keyword = 'CLS_features' if bert_type == 'CLS' else 'features'
    with open(tweetid_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            tweetid = line.strip()
            tweetid_list.append(tweetid)
    with open(bert_vec_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            content = json.loads(line.strip())
            bertvec = content[feat_keyword]['-{}'.format(layer)]
            tweetid2bertvec[tweetid_list[i]] = bertvec
    return tweetid2bertvec


def get_tweetid2vec_by_npy(tweetid_file: str, npy_vec_file: str) -> Dict[str, List[float]]:
    """
    Here the skip thought and the fasttext-crawl use .npy format to store the vectors
    :param tweetid_file:
    :param npy_vec_file:
    :return:
    """
    tweetid2vec = dict()
    feat_vectors = np.load(npy_vec_file)

    with open(tweetid_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            tweetid = line.strip()
            tweetid2vec[tweetid] = feat_vectors[i].tolist()
    return tweetid2vec


def extract_hand_crafted_feature(content_list: list) -> (np.ndarray, List[str]):
    """
    Part of the features are taken from Davidson et al.
    This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features
    :param content_list:
    :return:
    """
    feature_list = []
    clean_text_list = []
    for content in content_list:
        current_feature = []
        # Some statistical info provided by the official tweets API
        entities = content['entities']
        for feature_name in ['hashtags', 'symbols', 'user_mentions', 'urls', 'media']:
            if feature_name in entities:
                current_feature.append(len(entities[feature_name]))
            else:
                current_feature.append(0)
        current_feature.append(content["retweet_count"])
        current_feature.append(0 if content["retweet_count"] == 0 else 1)
        current_feature.append(content["favorite_count"])

        # Some info from the text content (after clean)
        text = get_clean_tweet(content['full_text'], entities)
        clean_text_list.append(text)
        words = tweet_tokenizer.tokenize(text)

        feat_name2val = dict()
        sentiment = sentiment_analyzer.polarity_scores(text)
        feat_name2val['sentiment_pos'] = sentiment['pos']
        feat_name2val['sentiment_neg'] = sentiment['neg']
        feat_name2val['sentiment_neu'] = sentiment['neu']
        feat_name2val['sentiment_compound'] = sentiment['compound']
        feat_name2val['num_chars'] = sum(len(w) for w in words)
        feat_name2val['num_chars_total'] = len(text)
        feat_name2val['num_terms'] = len(text.split())
        feat_name2val['num_words'] = len(words)
        feat_name2val['num_unique_terms'] = len(set([x.lower() for x in words]))
        feat_name2val['caps_count'] = sum([1 if x.isupper() else 0 for x in text])
        feat_name2val['caps_ratio'] = feat_name2val['caps_count'] / feat_name2val['num_chars_total']
        feat_name2val['has_place'] = 1 if "coordinates" in content and content['coordinates'] is not None else 0
        feat_name2val['is_verified'] = 1 if content['user']["verified"] else 0
        for feat_name in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_compound', 'num_chars',
                          'num_chars_total', 'num_terms', 'num_words', 'num_unique_terms', 'caps_count', 'caps_ratio',
                          'has_place', 'is_verified']:
            current_feature.append(feat_name2val[feat_name])
        # Some other features added after reviewing attributes in tweets and users
        current_feature.append(content['user']['favourites_count'])
        current_feature.append(content['user']['followers_count'])
        current_feature.append(content['user']['statuses_count'])
        current_feature.append(content['user']['geo_enabled'])
        current_feature.append(content['user']['listed_count'])
        current_feature.append(content['user']['friends_count'])

        feature_list.append(current_feature)

    feature_list = np.asarray(feature_list, dtype=np.float32)
    return feature_list, clean_text_list



def extract_cbnu_user_feature(content_list: list, annotated_user_type: dict) -> (np.ndarray):
    """
    Part of the features are taken from the CBNU paper
    This function takes a string and returns a list of length 6
    It indicates whether certain user entity types exist in the user mentions of the tweet 
    
    :param content_list:
    :param annotated_user_type:
    :return:
    """
    feature_list = []
    # clean_text_list = []
    for content in content_list:
        # initialize the feature list with zero
        # from position 0-5, the number indicates if the following user mention entities exist
        # news, weather, organization, donation, disaster information, multimedia
        current_feature = [0] * 6
        user_mentions = content['entities']['user_mentions']
        if len(user_mentions) > 0:
            for user_metion in user_mentions:
                if user_metion['id_str'] in annotated_user_type.keys():
                    mention_type = annotated_user_type['id_str']
                    if mention_type == 'news':
                        current_feature[0] = 1
                    elif mention_type == 'weather':
                        current_feature[1] = 1
                    elif mention_type == 'organization':
                        current_feature[2] = 1
                    elif mention_type == 'donation':
                        current_feature[3] = 1
                    elif mention_type == 'disaster':
                        current_feature[4] = 1
                    elif mention_type == 'multimedia':
                        current_feature[5] = 1
        feature_list.append(current_feature)

    feature_list = np.asarray(feature_list, dtype=np.float32)
 
    return feature_list


def get_clean_tweet(tweet: str, enities_info: dict):
    """
    Replace the text in hashtags, and remove the mentions and urls (indices are provided by twitter API)
    Todo: The hashtags may need to be taken care of, because it contains critical info
    :param tweet:
    :param enities_info:
    :return:
    """
    indices_txt = []
    for hashtag in enities_info['hashtags']:
        txt = hashtag['text']
        start, end = hashtag['indices']
        indices_txt.append([start, end, txt])
    for user_mention in enities_info['user_mentions']:
        txt = ''
        start, end = user_mention['indices']
        indices_txt.append([start, end, txt])
    for url in enities_info['urls']:
        txt = ''
        start, end = url['indices']
        indices_txt.append([start, end, txt])
    indices_txt = sorted(indices_txt, key=lambda x: x[0], reverse=True)
    for start_idx, end_idx, txt in indices_txt:
        tweet = tweet[: start_idx] + txt + tweet[end_idx:]
    # Some url is part of the content
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                 '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweet = re.sub(url_regex, '', tweet)
    # Remove continuous space
    space_pattern = '\s+'
    tweet = re.sub(space_pattern, ' ', tweet)
    return tweet


def print_to_log(content, level='info'):
    print_func = getattr(logger, level)
    print_func(content)


def extract_train_ids_to_file():
    """
    Extract all tweets ids in the training data, which will be used by the twarc to get the json content
    :return:
    """
    filepath = os.path.join('data', 'TRECIS-CTIT-H-Training.json')
    outfile = os.path.join('data', 'train-ids.txt')
    fout = open(outfile, 'w', encoding='utf8')
    with open(filepath, 'r', encoding='utf8') as f:
        content = json.load(f)
    tweet_ids = set()
    for events in content['events']:
        for tweets in events['tweets']:
            current_id = tweets['postID']
            if current_id in tweet_ids:
                print("Find duplicate tweets {0} in the {1}".format(current_id, filepath))
            else:
                tweet_ids.add(current_id)
                fout.write(current_id + '\n')
    fout.close()


def extract_test_ids_to_file():
    filepath = os.path.join('data', 'TRECIS-CTIT-H-Test.tweetids.tsv')
    outfile = os.path.join('data', 'test-ids.txt')
    fout = open(outfile, 'w', encoding='utf8')
    tweet_ids = set()
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            current_id = line.strip().split('\t')[2]
            if current_id in tweet_ids:
                print("Find duplicate tweets {0} in the {1}".format(current_id, filepath))
            else:
                tweet_ids = set()
                fout.write(current_id + '\n')
    fout.close()


def gzip_compress_file(filepath):
    """
    Gzip the file for evaluartion. Remove the former gz file if it exists
    :param filepath:
    :return:
    """
    import subprocess
    former_file = filepath + '.gz'
    if os.path.isfile(former_file):
        os.remove(former_file)
    subprocess.run(["gzip", filepath])


def find_true_relevant(event_id='bostonBombings2013'):
    for filepath in ['data/TRECIS-2018-TestEvents-Labels/assr{}.test'.format(i) for i in range(1, 7)]:
        with open(filepath, 'r', encoding='utf8') as f:
            content = json.load(f)
            event_in = False
            for event_info in content['annotator']['eventsAnnotated']:
                if event_id == event_info['identifier']:
                    event_in = True
                    break
            if not event_in:
                continue
            for event in content['events']:
                if event['eventid'] != event_id:
                    continue
                for tweet in event['tweets']:
                    if 'PastNews' in tweet['categories'] or 'Irrelevant' in tweet['categories']:
                        continue
                    print("{0}    {1}".format(tweet['postID'], tweet['categories']))


def rescale_score(score_list, proportion=0.5, threshold=0.7):
    """Rescale the score list to make at least `proportion` of data is >= `threshold`.

    For each score that falls in the larger proportion, we use (score - min_score) / (max_score - min_score) to first
    convert it to [0, 1], and then multiply it by (1 - threshold) to make it to [0, 1-threshold], and finally add it by
    threshold, to get the final range [threshold, 1].
    For each score that falls in the smaller proportion, we use the similar process to convert it to [0, threshold].

    :param score_list: A list of score which has been sorted in descending order.
    :param proportion: The proportion of scores that we want to make them larger than threshold.
    :param threshold: The threshold.

    :return: The score list after rescaled.
    """
    assert len(score_list) > 0
    assert score_list[0] > score_list[-1]
    score_num = len(score_list)
    above_threshold_num = int(proportion * score_num)

    above_threshold_min_score = score_list[above_threshold_num - 1]
    above_threshold_max_score = score_list[0]
    above_max_min_diff = above_threshold_max_score - above_threshold_min_score
    below_threshold_min_score = score_list[-1]
    below_threshold_max_score = above_threshold_min_score
    below_max_min_diff = below_threshold_max_score - below_threshold_min_score
    for idx, score in enumerate(score_list):
        if score < above_threshold_min_score:
            score_list[idx] = ((score - below_threshold_min_score) / below_max_min_diff) * threshold
        else:
            score_list[idx] = ((score - above_threshold_min_score) / above_max_min_diff) * (1 - threshold) + threshold
    return score_list


def rescale_and_write(content_list, f_out):
    score_list = [float(content[4]) for content in content_list]
    score_list = rescale_score(score_list, proportion=0.02)
    for idx, content in enumerate(content_list):
        content[4] = str(score_list[idx])
        f_out.write("\t".join(content) + "\n")


def get_rescale_file(model_name='rf'):
    input_file = os.path.join('eval', '{}.run'.format(model_name))
    out_file = os.path.join('eval', '{}-rescale.run'.format(model_name))
    content_list = []
    last_dataset_name = None
    f_out = open(out_file, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            content = line.strip().split('\t')
            dataset_name = content[0]
            if last_dataset_name is not None and dataset_name != last_dataset_name:
                rescale_and_write(content_list, f_out)
            content_list.append(content)
    rescale_and_write(content_list, f_out)
    f_out.close()


def get_data_size(filename_list):
    """ Get the size for each dataset for writing the paper. """
    data_size = 0
    for filename in filename_list:
        is_2019_data = '2019' in filename
        with open(filename, 'r', encoding='latin-1' if is_2019_data else 'utf8') as f:
            train_file_content = json.load(f)
            for event in train_file_content['events']:
                data_size += len(event['tweets'])
    return data_size


if __name__ == '__main__':
    data_dir = 'data'
    train_file_list = []
    train_file_list += [os.path.join(data_dir, '2019ALabels', '2019A-assr{}.json'.format(i)) for i in range(1, 6)]
    train_file_list += [os.path.join(data_dir, '2019ALabels', '2019-assr2.json')]
    print("The 2019-A test size is {}".format(get_data_size(train_file_list)))
