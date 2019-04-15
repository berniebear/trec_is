import os
from typing import List
import argparse
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Input file contains sentence one per line")
    parser.add_argument("--output_file", type=str,
                        help="The feature output by the glove model")
    return parser.parse_args()


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


def extract_by_glove(sent_list: List[str]) -> np.ndarray:
    glove_path = os.path.join('../../../data/glove.twitter.27B.200d.txt')
    word2vecs, vec_dim = load_glove(glove_path)

    features = []
    count_miss, count_total = 0, 0
    for sent in sent_list:
        sent_feat = []
        for word in sent.split():
            count_total += 1
            if word in word2vecs:
                sent_feat.append(word2vecs[word])
            else:
                count_miss += 1
        if len(sent_feat) == 0:
            features.append(np.zeros([vec_dim], dtype=np.float))
        else:
            features.append(np.mean(np.asarray(sent_feat), axis=0))

    print("There are {0}/{1} words missed by the glove in tweets".format(count_miss, count_total))
    return np.asarray(features)


def main():
    args = get_arguments()

    sent_list = []
    empty_sent = total_sent = 0
    with open(args.input_file, 'r', encoding='utf8') as f:
        for line in f:
            total_sent += 1
            line = line.strip()
            if len(line) == 0:
                line = '<PAD>'
                empty_sent += 1
            sent_list.append(line)
    print("There are {0}/{1} empty lines in {2}".format(empty_sent, total_sent, args.input_file))
    vectors = extract_by_glove(sent_list)

    np.save(args.output_file, vectors)
    print("The feature vectors has been written to {}".format(args.output_file))


if __name__ == '__main__':
    main()

