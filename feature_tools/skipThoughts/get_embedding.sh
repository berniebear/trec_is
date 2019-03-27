#!/usr/bin/env bash

nohup python -u extract_feature.py \
  --input_file=../../out/tweets-clean-text.txt \
  --output_file=../../out/skip-thought-vec.npy \
  >~/log/Mar27.log 2>&1 &
