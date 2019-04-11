import os
import re
import pickle
import logging
import json
from typing import List, Dict
import numpy as np
from scipy.sparse import csr_matrix

# For processing tweets
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.fasttext import FastText
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
porter = PorterStemmer()
tweet_tokenizer = TweetTokenizer()
sentiment_analyzer = SentimentIntensityAnalyzer()

logger = logging.getLogger()


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


def formalize_train_file(origin_train_file: str, formalized_train_file: str):
    fout = open(formalized_train_file, 'w', encoding='utf8')
    formalize_helper(origin_train_file, fout)
    fout.close()


def formalize_test_file(origin_test_files: List[str], formalized_test_file: str):
    fout = open(formalized_test_file, 'w', encoding='utf8')
    for test_file in origin_test_files:
        formalize_helper(test_file, fout)
    fout.close()


def formalize_helper(filename, fout):
    with open(filename, 'r', encoding='utf8') as f:
        train_file_content = json.load(f)
        for event in train_file_content['events']:
            for tweet in event['tweets']:
                tweetid = tweet['postID']
                label_list = tweet['categories']
                priority = tweet['priority']
                fout.write('{0}\t{1}\t{2}\n'.format(tweetid, ','.join(label_list), priority))


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
    for content in tweet_content_list:
        entities = content['entities']
        clean_text = get_clean_tweet(content['full_text'], entities)
        clean_texts.append(clean_text)
    write_list_to_file(clean_texts, tweet_text_out_file)
    write_list_to_file(tweetid_list, tweet_id_out_file)
    print("The tweet text and ids has been written to {0} and {1}".format(tweet_text_out_file, tweet_id_out_file))


def get_label2id(label_file: str, train_file: str, threshold: int):
    """
    Because the original labels are in the form of text, such as "MultimediaShare" and so on
    We want to convert those textual labels to digital labels
    :param label_file: All types of labels provided by TREC, including explanation of each type of label
    :param train_file: Formalized train file, where each line is in the form of "{tweetid}\t{labels}\t{Priority}\n"
    :param threshold:
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
    return label2id, majority_label, short2long_label


def get_id2label(label2id: dict):
    id2label = [''] * len(label2id)
    for label, idx in label2id.items():
        id2label[idx] = label
    return id2label


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


def get_tweetid2vec(tweetid_file: str, vec_file: str, feat_name: str) -> Dict[str, List[float]]:
    """
    The file of tweets content is args.tweet_text_out_file
    The file of corresponding feature vector (such as bert or skip-thought) is vec_file
    The file of tweets id is tweetid_file
    :param tweetid_file:
    :param vec_file:
    :param feat_name:
    :return:
    """
    if feat_name == 'bert':
        return get_tweetid2bertvec(tweetid_file, vec_file)
    elif feat_name == 'skip-thought':
        return get_tweetid2skipthought_vec(tweetid_file, vec_file)
    else:
        tweetid2vec = dict()
        tweetid_list = []
        with open(tweetid_file, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                tweetid = line.strip()
                tweetid_list.append(tweetid)
        with open(vec_file, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                tweetid2vec[tweetid_list[i]] = json.loads(line.strip())
        return tweetid2vec


def get_tweetid2bertvec(tweetid_file: str, bert_vec_file: str) -> Dict[str, List[float]]:
    tweetid2bertvec = dict()
    tweetid_list = []
    with open(tweetid_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            tweetid = line.strip()
            tweetid_list.append(tweetid)
    with open(bert_vec_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            content = json.loads(line.strip())
            bertvec = content['features']['-1']
            tweetid2bertvec[tweetid_list[i]] = bertvec
    return tweetid2bertvec


def get_tweetid2skipthought_vec(tweetid_file: str, skipthought_vec_file: str) -> Dict[str, List[float]]:
    """
    Here the skip thought use .npy format to store the vectors
    :param tweetid_file:
    :param skipthought_vec_file:
    :return:
    """
    tweetid2skipthought_vec = dict()
    feat_vectors = np.load(skipthought_vec_file)

    with open(tweetid_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            tweetid = line.strip()
            tweetid2skipthought_vec[tweetid] = feat_vectors[i].tolist()
    return tweetid2skipthought_vec


def extract_by_tfidf(texts: list, vectorizer: TfidfVectorizer) -> csr_matrix:
    return vectorizer.transform(texts)


def extract_by_fasttext(texts: list, vectorizer, analyzer,
                        tfidf_feature: csr_matrix, tfidf_vocab: list, merge: str = 'avg'):
    return extract_by_word_embed(texts, vectorizer, analyzer, tfidf_feature, tfidf_vocab, 'fasttext', merge)


def extract_by_glove(texts: list, vectorizer, analyzer,
                     tfidf_feature: csr_matrix, tfidf_vocab: list, merge: str = 'avg'):
    return extract_by_word_embed(texts, vectorizer, analyzer, tfidf_feature, tfidf_vocab, 'glove', merge)


def extract_by_word_embed(texts: list, vectorizer, analyzer,
                          tfidf_feature: csr_matrix, tfidf_vocab: list, embed_name: str, merge: str):
    """
    We get two kinds of fasttext features here, the first is the simple average,
        the other is weighted sum according to tfidf score.
    tfidf_feature is [sent_num, vocab_size], and by the fasttext we can get feature for the vocab [vocab_size, embed_dim]
    Then we can use [sent_num, vocab_size] * [vocab_size, embed_dim] to get a weighted sum of [sent_num, embed_dim] feature
    :param texts:
    :param vectorizer:
    :param analyzer:
    :param tfidf_feature:
    :param tfidf_vocab:
    :param embed_name: fasttext or glove, here we provide a uniform API for it
    :param merge: The type of merge for tokens to get sentence feature. 'avg' means average merging,
            'weighted' means weighted sum according to tf-idf weight
    :return:
    """
    if merge == 'avg':
        # Get the simple average feature
        avg_feature = []
        count_miss = 0
        for sentence in texts:
            tokenized = [normalize_for_fasttext(t) for t in analyzer(sentence)]
            wvs = []
            for t in tokenized:
                try:
                    token_vec = vectorizer.wv[t]
                    # norm = np.linalg.norm(token_vec)
                    # normed_token_vec = token_vec / norm
                    wvs.append(token_vec)
                except KeyError:
                    wvs.append(np.zeros([vectorizer.vector_size], dtype=np.float32))
                    count_miss += 1
            if len(wvs) == 0:
                sentence_vec = np.zeros([vectorizer.vector_size])
            else:
                sentence_vec = np.mean(np.asarray(wvs), axis=0)
            avg_feature.append(sentence_vec)
        fasttext_feature = np.asarray(avg_feature)  # [sent_num, embed_dim]
        assert len(fasttext_feature.shape) == 2, \
            "The shape for {0} of avg_feature is {1}".format(embed_name, fasttext_feature.shape)
        print_to_log("There are {0} words missed by the {1} in tweets".format(count_miss, embed_name))

    elif merge == 'weighted':
        # Get the weighted sum feature by tf-idf score
        count_miss = 0
        fasttext_for_vocab = []
        for word in tfidf_vocab:
            try:
                fasttext_vec = vectorizer.wv[word]
            except KeyError:
                fasttext_vec = np.zeros([vectorizer.vector_size], dtype=np.float32)
                count_miss += 1
            fasttext_for_vocab.append(fasttext_vec)
        fasttext_for_vocab = np.asarray(fasttext_for_vocab)
        fasttext_feature = tfidf_feature.dot(fasttext_for_vocab)
        print_to_log("There are {0}/{1} words missed by the {2} in tfidf vocab".format(
            count_miss, tfidf_feature.shape[1], embed_name))

    else:
        raise ValueError("The value of merge {} is invalid".format(merge))

    return fasttext_feature


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

        feature_list.append(current_feature)
    feature_list = np.asarray(feature_list, dtype=np.float32)
    return feature_list, clean_text_list


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


def normalize_for_fasttext(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    From: https://www.kaggle.com/mschumacher/using-fasttext-models-for-robust-embeddings
    """
    s = s.lower()

    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', 'zero')
    s = s.replace('1', 'one')
    s = s.replace('2', 'two')
    s = s.replace('3', 'three')
    s = s.replace('4', 'four')
    s = s.replace('5', 'five')
    s = s.replace('6', 'six')
    s = s.replace('7', 'seven')
    s = s.replace('8', 'eight')
    s = s.replace('9', 'nine')

    return s


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


if __name__ == '__main__':
    find_true_relevant()
