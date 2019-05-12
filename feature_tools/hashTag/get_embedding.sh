#!/usr/bin/env bash

python -u extract_hashtag2vec.py \
  --input_file=../../out/tweets-id.txt \
  --output_file=../../out/hashtag-vec.npy

python -u extract_hashtag2vec.py \
  --input_file=../../out/tweets-id-2019.txt \
  --output_file=../../out/hashtag-vec-2019.npy
