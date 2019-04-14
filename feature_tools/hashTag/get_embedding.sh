#!/usr/bin/env bash

nohup python -u extract_hashtag2vec.py \
  --input_file=../../out/tweets-id.txt \
  --output_file=../../out/hashtag-vec.npy \
  $@