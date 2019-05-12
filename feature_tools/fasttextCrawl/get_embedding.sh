#!/usr/bin/env bash

python -u extract_by_fasttext.py \
  --input_file=../../out/tweets-clean-text.txt \
  --output_file=../../out/fasttext-crawl-vec.npy

python -u extract_by_fasttext.py \
  --input_file=../../out/tweets-clean-text-2019.txt \
  --output_file=../../out/fasttext-crawl-vec-2019.npy
