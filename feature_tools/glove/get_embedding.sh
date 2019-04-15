#!/usr/bin/env bash

nohup python -u extract_by_glove.py \
  --input_file=../../out/tweets-clean-text.txt \
  --output_file=../../out/glove-vec.npy \
  $@