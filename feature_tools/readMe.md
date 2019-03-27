## How to use

All those tools are external tools to generate features.
The input format should be a file, where each line is a sentence, and the tool will generate a file where each line contains the info about the feature of that sentence.

### BERT
https://github.com/google-research/bert#using-bert-to-extract-fixed-feature-vectors-like-elmo

We first store all sentences to a file, and then use the BERT API to store the result json to a file.
Then read that json file and process it to get the vector of this.

1. Clone the official bert code into `./bert`
1. Download the pretrained model into `./uncased_L-24_H-1024_A-16`
1. Set the `input_file` and `output_file` in `get_embedding.sh`
1. Run the `bash get_embedding.sh`

Tips: Can run on Google Colab because it only needs several hours

### Skip-thoughts


Here we use the skip-thought code implemented by TensorFlow Examples [here](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)

1. Clone the official bert code into `./skip_thoughts`
1. Download the pretrained model
1. Set the `skip_thought_dir` in `extract_feature.py`
1. Set the `input_file` and `output_file` in `get_embedding.sh`
1. Run the `bash get_embedding.sh`
