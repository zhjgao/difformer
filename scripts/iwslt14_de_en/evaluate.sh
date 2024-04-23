#!/bin/bash

STEPS=20

LENGTH_BEAM=7
NOISE_BEAM=3

DATASET="iwslt14_de_en"
MODEL_DIR="models/${DATASET}"

OUTPUT_NAME="evaluate_step${STEPS}_beam${LENGTH_BEAM}x${NOISE_BEAM}"
OUTPUT_DIR=$MODEL_DIR/$OUTPUT_NAME

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 fairseq-generate \
    data-bin/${DATASET}_distill \
    --gen-subset test \
    --user-dir difformer \
    --task difformer \
    --path $MODEL_DIR/difformer.pt:$MODEL_DIR/transformer.pt \
    --decoding-steps $STEPS \
    --decoding-early-stopping 5 \
    --length-beam-size $LENGTH_BEAM \
    --noise-beam-size $NOISE_BEAM \
    --ppl-mbr \
    --remove-bpe \
    --batch-size 50 \
    > $OUTPUT_DIR/output.txt

tail -n 1 $OUTPUT_DIR/output.txt > $OUTPUT_DIR/bleu.txt

echo "Finished $OUTPUT_NAME. BLEU:"
cat $OUTPUT_DIR/bleu.txt
