## How to use

All those tools are external tools to generate features.
The input format should be a file, where each line is a sentence, and the tool will generate a file where each line contains the info about the feature of that sentence.

If you want to extract all features, the simple usage is running
```bash
bash extract_features.sh
```

### BERT
https://github.com/google-research/bert#using-bert-to-extract-fixed-feature-vectors-like-elmo

We first store all sentences to a file, and then use the BERT API to store the result json to a file.
Then read that json file and process it to get the vector of this.

1. Clone the official bert code into `./bert`
1. Download the pretrained model into `./uncased_L-24_H-1024_A-16`
1. Set the `input_file` and `output_file` in `get_embedding.sh`
1. Run the `bash get_embedding.sh`

Tips: 
- Can run on Google Colab because it only needs several hours.
- If you want to get some other embeddings (such as the embedding combined by tf-idf), you can edit the code around line 400 in `extract_sent_features.py`

### Skip-thoughts

Here we use the skip-thought code implemented by TensorFlow Examples [here](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)

1. Clone the official bert code into `./skip_thoughts`
1. Download the pretrained model
1. Set the `skip_thought_dir` in `extract_feature.py`
1. Set the `input_file` and `output_file` in `get_embedding.sh`
1. Run the `bash get_embedding.sh`

### gloveAndFasttext

Use glove and fasttext (provided by TREC host trained on 1M tweets) to get the embedding.
It will generate four embedding files, including the glove/fasttext + use avg/tf-idf to do the combination.

1. Direct run the `bash get_embedding.sh`

### FastTextCrawl

Although the official host provides the fasttext pretrained on the 2013-2016 twitter data, we can still use another pretrained provided by Facebook,
which is 2 million word vectors trained with subword information on Common Crawl (600B tokens).

1. Download the pretrained fasttext model from [here](https://fasttext.cc/docs/en/english-vectors.html)
1. If we only want to use the static file, we can just find all words in the .vec file
1. However, as we want to utilize the subword info to cope with the OOV, we need to use the .bin file
1. Just run the `bash get_embedding.sh`

### hashTag

Use the Fasttext to get the embedding of each hashTag. First we collect all hashtags, then run the fasttext on it and get hashtag2vec.
Then we can process all tweets. If a tweet contains more than one hashtags, we choose to average the none-zero vectors of it.

1. Here we use the fasttext train on tweets, which is provided by the host
1. Direct run the `bash get_embedding.sh`
