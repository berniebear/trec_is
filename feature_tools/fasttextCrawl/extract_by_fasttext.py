import os
from typing import List
import argparse
import fastText
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Input file contains sentence one per line")
    parser.add_argument("--output_file", type=str,
                        help="The feature output by the skip thought model")
    return parser.parse_args()


def extract_by_fasttext(sent_list: List[str]) -> np.ndarray:
    model_path = os.path.join("../../../data", "fasttextCommonCrawl", "crawl-300d-2M-subword.bin")
    model = fastText.load_model(model_path)
    vector_size = model.get_dimension()

    avg_feature = []
    count_miss = 0
    for sentence in sent_list:
        sentence_vec = model.get_sentence_vector(sentence)
        avg_feature.append(sentence_vec)
    fasttext_feature = np.asarray(avg_feature)  # [sent_num, embed_dim]
    assert len(fasttext_feature.shape) == 2
    print("There are {0} words missed by the fasttext crawl in tweets".format(count_miss))
    return fasttext_feature


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
    vectors = extract_by_fasttext(sent_list)

    np.save(args.output_file, vectors)
    print("The feature vectors has been written to {}".format(args.output_file))


if __name__ == '__main__':
    main()

