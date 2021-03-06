import os
from typing import List, Dict
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from nltk.tokenize import TweetTokenizer
local_tokenizer = TweetTokenizer()


class GloveVectorizer(object):
    def __init__(self, word2vecs: Dict[str, List[float]], vector_size: int):
        self.wv = word2vecs
        self.vector_size = vector_size


def tokenizer_wrapper(text):
    return local_tokenizer.tokenize(text)


def extract_by_tfidf(texts: List[str], vectorizer: TfidfVectorizer) -> csr_matrix:
    return vectorizer.transform(texts)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Input file contains sentence one per line")
    parser.add_argument("--output_dir", type=str,
                        help="The feature output directory (because we will extract multiple models in this script)")
    return parser.parse_args()


def get_tfidf_vectorizer(tfidf_model_path) -> TfidfVectorizer:
    return joblib.load(tfidf_model_path)


def load_glove(glove_path: str):
    word2vecs = dict()
    vec_dim = None
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            embedding = list(map(float, line[1:]))
            if vec_dim is None:
                vec_dim = len(embedding)
            word2vecs[word] = embedding
    return word2vecs, vec_dim


def read_clean_texts(input_file):
    sent_list = []
    empty_sent = total_sent = 0
    with open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            total_sent += 1
            line = line.strip()
            if len(line) == 0:
                line = '<PAD>'
                empty_sent += 1
            sent_list.append(line)
    assert empty_sent == 0, "There are {0}/{1} empty lines in {2}".format(empty_sent, total_sent, input_file)
    return sent_list


def get_glove_vectorizer(glove_path: str) -> GloveVectorizer:
    word2vecs = dict()
    vec_dim = None
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            embedding = [float(val) for val in line[1:]]
            if vec_dim is None:
                vec_dim = len(embedding)
            word2vecs[word] = embedding
    return GloveVectorizer(word2vecs, vec_dim)


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


def extract_by_fasttext(texts: List[str], vectorizer, analyzer,
                        tfidf_feature: csr_matrix, tfidf_vocab: list, merge: str = 'avg'):
    return extract_by_word_embed(texts, vectorizer, analyzer, tfidf_feature, tfidf_vocab, 'fasttext', merge)


def extract_by_glove(texts: List[str], vectorizer, analyzer,
                     tfidf_feature: csr_matrix, tfidf_vocab: list, merge: str = 'avg'):
    return extract_by_word_embed(texts, vectorizer, analyzer, tfidf_feature, tfidf_vocab, 'glove', merge)


def extract_by_word_embed(texts: List[str], vectorizer, analyzer,
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
        count_miss, count_total = 0, 0
        for sentence in texts:
            tokenized = [normalize_for_fasttext(t) for t in analyzer(sentence)]
            wvs = []
            for t in tokenized:
                count_total += 1
                try:
                    token_vec = vectorizer.wv[t]
                    wvs.append(token_vec)
                except KeyError:
                    count_miss += 1
            if len(wvs) == 0:
                sentence_vec = np.zeros([vectorizer.vector_size])
            else:
                sentence_vec = np.mean(np.asarray(wvs), axis=0)
            avg_feature.append(sentence_vec)
        feature = np.asarray(avg_feature)  # [sent_num, embed_dim]
        assert len(feature.shape) == 2, "The shape for {0} of avg_feature is {1}".format(embed_name, feature.shape)
        print("There are {0}/{1} words missed by the {2} in tweets".format(count_miss, count_total, embed_name))

    elif merge == 'weighted':
        # Get the weighted sum feature by tf-idf score
        count_miss = 0
        embed_for_vocab = []
        for word in tfidf_vocab:
            try:
                fasttext_vec = vectorizer.wv[word]
            except KeyError:
                fasttext_vec = np.zeros([vectorizer.vector_size], dtype=np.float32)
                count_miss += 1
            embed_for_vocab.append(fasttext_vec)
        embed_for_vocab = np.asarray(embed_for_vocab)
        tfidf_feature = normalize(tfidf_feature, norm='l1')
        feature = tfidf_feature.dot(embed_for_vocab)  # [sent_num, vocab_size] * [vocab_size, embed_dim]
        print("feature shape is {0}, embed_for_vocab shape is {1}, tfidf_feature shape is {2}".format(
            feature.shape, embed_for_vocab.shape, tfidf_feature.shape))
        print("There are {0}/{1} words missed by the {2} in tfidf vocab".format(
            count_miss, tfidf_feature.shape[1], embed_name))

    else:
        raise ValueError("The value of merge {} is invalid".format(merge))

    return feature


def get_fasttext_vectorizer(fasttext_model_path):
    from gensim.models.fasttext import FastText
    return FastText.load(fasttext_model_path)


def main():
    args = get_arguments()

    tf_idf_path = '../../../data/2013to2016_tfidf_vectorizer_20190109.pkl'
    fasttext_model_path = '../../../data/text_sample_2013to2016_gensim_200.model'
    glove_path = os.path.join('../../../data/glove.twitter.27B.200d.txt')

    tfidf_vectorizer = get_tfidf_vectorizer(tf_idf_path)
    analyzer = tfidf_vectorizer.build_analyzer()
    clean_texts = read_clean_texts(args.input_file)
    tfidf_feature = extract_by_tfidf(clean_texts, tfidf_vectorizer)

    glove_vectorizer = get_glove_vectorizer(glove_path)
    glove_feature_avg = extract_by_glove(clean_texts, glove_vectorizer, analyzer,
                                         tfidf_feature, tfidf_vectorizer.get_feature_names(), 'avg')
    glove_feature_tfidf = extract_by_glove(clean_texts, glove_vectorizer, analyzer,
                                           tfidf_feature, tfidf_vectorizer.get_feature_names(), 'weighted')

    fasttext_vectorizer = get_fasttext_vectorizer(fasttext_model_path)
    fasttext_feature_avg = extract_by_fasttext(clean_texts, fasttext_vectorizer, analyzer,
                                               tfidf_feature, tfidf_vectorizer.get_feature_names(), 'avg')
    fasttext_feature_tfidf = extract_by_fasttext(clean_texts, fasttext_vectorizer, analyzer,
                                                 tfidf_feature, tfidf_vectorizer.get_feature_names(), 'weighted')

    postfix = "-2019" if args.input_file.endswith("2019.txt") else ""
    np.save(os.path.join(args.output_dir, 'glove-avg-vec{}.npy'.format(postfix)), glove_feature_avg)
    np.save(os.path.join(args.output_dir, 'glove-tfidf-vec{}.npy'.format(postfix)), glove_feature_tfidf)
    np.save(os.path.join(args.output_dir, 'fasttext-avg-vec{}.npy'.format(postfix)), fasttext_feature_avg)
    np.save(os.path.join(args.output_dir, 'fasttext-tfidf-vec{}.npy'.format(postfix)), fasttext_feature_tfidf)
    print("The feature vectors has been written to {}".format(args.output_dir))


if __name__ == '__main__':
    main()
