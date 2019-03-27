import os
import argparse
import numpy as np
from typing import List

from skip_thoughts import configuration
from skip_thoughts import encoder_manager


def get_arguments():
    parser = argparse.ArgumentParser()
    # sanity check
    parser.add_argument("--input_file", type=str,
                        help="Input file contains sentence one per line")
    parser.add_argument("--output_file", type=str,
                        help="The feature output by the skip thought model")
    return parser.parse_args()


def extract_by_skip_thought(sent_list: List[str]):
    """
    To make it compatible with the toolkit, we need the input to be a list of sentences
    :param sent_list:
    :return:
    """
    skip_thought_dir = os.path.join('/home/junpeiz/Project/Twitter/data', 'skipThoughts', 'pretrained', 'skip_thoughts_uni_2017_02_02')
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
    vectors = extract_by_skip_thought(sent_list)

    np.save(args.output_file, vectors)
    print("The feature vectors has been written to {}".format(args.output_file))


if __name__ == '__main__':
    main()
