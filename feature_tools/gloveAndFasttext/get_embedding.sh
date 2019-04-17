#!/usr/bin/env bash

nohup python -u extract_by_glove_fasttext.py \
  --input_file=../../out/tweets-clean-text.txt \
  --output_dir=../../out \
  $@