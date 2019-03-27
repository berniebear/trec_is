import argparse
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
    skip_thought_dir = os.path.join('../data', 'skipThoughts', 'pretrained', 'skip_thoughts_uni_2017_02_02')
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
    with open(args.input_file, 'r', encoding='utf8') as f:
        for line in f:
            sent_list.append(line.strip())
    vector_list = extract_by_skip_thought(sent_list)

    with open(args.output_file, 'w', encoding='utf8') as f:
        for vector in vector_list:
            f.write('{}\n'.format(vector))

    print("The feature vectors has been written to {}".format(args.output_file))


if __name__ == '__main__':
    main()
