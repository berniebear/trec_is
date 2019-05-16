#!/usr/bin/env bash

python -u extract_by_fasttext.py \
  --input_file=../../out/event_1M_balanced_tweets.txt \
  --output_file=../../out/fasttext-1M-balanced-event-vec.npy

python -u extract_by_fasttext.py \
  --input_file=../../out/tweets-clean-text-2019.txt \
  --output_file=../../out/fasttext-1M-balanced-event-vec-2019.npy
