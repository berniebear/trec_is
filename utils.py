import os
import re
import pickle
import logging
import json
import numpy as np

# For processing tweets
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.fasttext import FastText
import preprocessor as p
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
porter = PorterStemmer()
tweet_tokenizer = TweetTokenizer()
sentiment_analyzer = SentimentIntensityAnalyzer()

from skip_thoughts import configuration
from skip_thoughts import encoder_manager


logger = logging.getLogger()


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


def extract_feature(content_list: list, tfidf_vectorizer: TfidfVectorizer, fasttext_vectorizer: FastText):
    analyzer = tfidf_vectorizer.build_analyzer()
    hand_crafted_feature, clean_texts = extract_hand_crafted_feature(content_list)
    tfidf_feature = extract_by_tfidf(clean_texts, tfidf_vectorizer)
    fasttext_feature = extract_by_fasttext(clean_texts, fasttext_vectorizer, analyzer)
    # skip_thought_feature = utils.extract_by_skip_thought(clean_texts)
    # bert_feature = utils.extract_by_bert(clean_texts)
    print_to_log("The shape of hand_crafted_feature is {}".format(hand_crafted_feature.shape))
    print_to_log("The shape of tfidf_feature is {}".format(tfidf_feature.shape))
    print_to_log("The shape of fasttext_feature is {}".format(fasttext_feature.shape))
    # All those features are [sent_num, feature_dim] size
    data_x = np.concatenate([hand_crafted_feature, tfidf_feature, fasttext_feature], axis=1)
    return data_x


def get_label2id(label_file, train_file, threshold):
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

    # Count label frequency
    label_count = {label: 0 for label in id2label}
    with open(train_file, 'r', encoding='utf8') as f:
        train_file_content = json.load(f)
        for event in train_file_content['events']:
            for tweet in event['tweets']:
                for tweet_label in tweet['categories']:
                    label_count[tweet_label] += 1
    print_to_log("All labels in {0} is: {1}".format(label_file, id2label))

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


def get_tweetid2content(tweet_file_list):
    tweetid2content = dict()
    for filename in tweet_file_list:
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                content = json.loads(line.strip())
                tweetid = content['id_str']
                tweetid2content[tweetid] = content
    return tweetid2content


def extract_by_skip_thought(args, sent_list):
    """
    To make it compatible with the toolkit, we need the input to be a list of sentences
    :param sent_list:
    :return:
    """
    skip_thought_dir = os.path.join(args.data_dir, 'skipThoughts', 'pretrained', 'skip_thoughts_uni_2017_02_02')
    # Set paths to the model.
    VOCAB_FILE = os.path.join(skip_thought_dir, "vocab.txt")
    EMBEDDING_MATRIX_FILE = os.path.join(skip_thought_dir, "embeddings.npy")
    CHECKPOINT_PATH = os.path.join(skip_thought_dir, "model.ckpt-501424")
    # The following directory should contain files rt-polarity.neg and
    # rt-polarity.pos.
    # MR_DATA_DIR = "/dir/containing/mr/data"

    # Set up the encoder. Here we are using a single unidirectional model.
    # To use a bidirectional model as well, call load_model() again with
    # configuration.model_config(bidirectional_encoder=True) and paths to the
    # bidirectional model's files. The encoder will use the concatenation of
    # all loaded models.
    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(),
                       vocabulary_file=VOCAB_FILE,
                       embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                       checkpoint_path=CHECKPOINT_PATH)

    encoding_list = encoder.encode(sent_list)
    return encoding_list


def extract_by_tfidf(texts: list, vectorizer: TfidfVectorizer):
    return vectorizer.transform(texts).toarray()


def extract_by_fasttext(texts: list, vectorizer: FastText, analyzer):
    feature_list = []
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
        feature_list.append(sentence_vec)
    print_to_log("There are {} words missed by the fasttext model".format(count_miss))
    return np.asarray(feature_list)


def extract_hand_crafted_feature(content_list: list):
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
        feat_name2val['has_place'] = 1 if "coordinates" in content else 0
        feat_name2val['is_verified'] = 1 if content['user']["verified"] else 0
        for feat_name in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_compound', 'num_chars',
                          'num_chars_total', 'num_terms', 'num_words', 'num_unique_terms', 'caps_count', 'caps_ratio',
                          'has_place', 'is_verified']:
            current_feature.append(feat_name2val[feat_name])

        feature_list.append(current_feature)
    feature_list = np.asarray(feature_list, dtype=np.float32)
    return feature_list, clean_text_list


def get_clean_tweet(tweet: str, enities_info: dict):
    # Replace the text in hashtags, and remove the mentions and urls (indices are provided by twitter API)
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

if __name__ == '__main__':
    gzip_compress_file(os.path.join('out', 'test.txt'))
