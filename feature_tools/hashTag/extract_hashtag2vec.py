import json
import argparse
import numpy as np
from gensim.models.fasttext import FastText


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Input file contains tweetid one per line")
    parser.add_argument("--output_file", type=str,
                        help="The feature output by the extract hashtag feature model")
    return parser.parse_args()


def collect_all_hashtags(test=False):
    hashtags_collect = set()
    tweetid2hashtags = dict()
    all_tweets_file = '../../data/all-tweets.txt' if not test else '../../data/all-tweets-2019.txt'
    with open(all_tweets_file, 'r', encoding='utf8') as f:
        for line in f:
            content = json.loads(line.strip())
            entities = content['entities']
            tweetid = content['id_str']
            for hashtag in entities['hashtags']:
                hashtags_collect.add(hashtag['text'])
            tweetid2hashtags[tweetid] = [hashtag['text'] for hashtag in entities['hashtags']]
    return hashtags_collect, tweetid2hashtags


def get_hashtag2vec(hashtags_collect, fasttext_model_path):
    fasttext = FastText.load(fasttext_model_path)
    hashtag2vec = dict()
    count_miss = 0
    for hashtag in hashtags_collect:
        try:
            token_vec = fasttext.wv[hashtag]
            hashtag2vec[hashtag] = token_vec
        except KeyError:
            count_miss += 1
    print("There are {0}/{1} hashtags that cannot get vector from fasttext".format(count_miss, len(hashtags_collect)))
    return hashtag2vec, fasttext.vector_size


def get_feature_by_tweet_id_file(feat_dim: int, hashtag2vec, tweetid2hashtags, tweetid_file) -> np.ndarray:
    """
    Notice that some hashtag may not have vectors, which means they are not in hashtag2vec
    :param feat_dim:
    :param hashtag2vec:
    :param tweetid2hashtags:
    :param tweetid_file:
    :return:
    """
    vectors = []
    with open(tweetid_file, 'r', encoding='utf8') as f:
        for line in f:
            tweetid = line.strip()
            hashtags = tweetid2hashtags[tweetid]
            feat_list = [hashtag2vec[hashtag] for hashtag in hashtags if hashtag in hashtag2vec]
            if len(feat_list) == 0:
                current_vector = np.zeros([feat_dim], dtype=np.float)
            else:
                current_vector = np.mean(np.asarray(feat_list), axis=0)
            vectors.append(current_vector)
    return np.asarray(vectors)


def main():
    args = get_arguments()
    fasttext_model_path = '../../../data/text_sample_2013to2016_gensim_200.model'
    hashtags_collect, tweetid2hashtags = collect_all_hashtags(test=args.input_file.endswith("2019.txt"))
    hashtag2vec, feat_dim = get_hashtag2vec(hashtags_collect, fasttext_model_path)
    vectors = get_feature_by_tweet_id_file(feat_dim, hashtag2vec, tweetid2hashtags, args.input_file)
    np.save(args.output_file, vectors)


if __name__ == '__main__':
    main()
