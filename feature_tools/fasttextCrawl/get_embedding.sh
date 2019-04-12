#!/usr/bin/env bash

nohup python -u extract_by_fasttext.py \
  --input_file=../../out/tweets-clean-text.txt \
  --output_file=../../out/fasttext-crawl-vec.npy \
  >~/log/Apr11.log 2>&1 &
