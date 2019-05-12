#!/usr/bin/env bash

python -u extract_feature.py \
  --input_file=../../out/tweets-clean-text.txt \
  --output_file=../../out/skip-thought-vec.npy

python -u extract_feature.py \
  --input_file=../../out/tweets-clean-text-2019.txt \
  --output_file=../../out/skip-thought-vec-2019.npy
