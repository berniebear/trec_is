export BERT_BASE_DIR=uncased_L-24_H-1024_A-16

# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.
# echo 'This is a test sentence .' > test_sent.txt
# echo 'Another line sentence ?' >> test_sent.txt

python bert/extract_sent_features.py \
  --input_file=tweets-clean-text.txt \
  --output_file=tweets-clean-text.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2 \
  --max_seq_length=128 \
  --batch_size=8
