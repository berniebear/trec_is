#!/usr/bin/env bash

python -u extract_by_glove_fasttext.py \
  --input_file=../../out/tweets-clean-text.txt \
  --output_dir=../../out

python -u extract_by_glove_fasttext.py \
  --input_file=../../out/tweets-clean-text-2019.txt \
  --output_dir=../../out
