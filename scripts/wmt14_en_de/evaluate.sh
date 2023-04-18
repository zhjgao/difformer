#!/bin/bash

STEPS=20

LENGTH_BEAM=7
NOISE_BEAM=3

DATASET="wmt14_en_de"
MODEL_DIR="models/${DATASET}"

OUTPUT_NAME="evaluate_step${STEPS}_beam${LENGTH_BEAM}x${NOISE_BEAM}"
OUTPUT_DIR=$MODEL_DIR/$OUTPUT_NAME

mkdir -p $OUTPUT_DIR/tmp

CUDA_VISIBLE_DEVICES=0 fairseq-generate \
    data-bin/${DATASET}_distill \
    --gen-subset test \
    --user-dir difformer \
    --task difformer \
    --path $MODEL_DIR/difformer.pt:$MODEL_DIR/transformer.pt \
    --decoding-steps $STEPS \
    --decoding-early-stopping 5 \
    --decoding-noise-schedule sqrt \
    --decoding-noise-factor 1 \
    --length-beam-size $LENGTH_BEAM \
    --noise-beam-size $NOISE_BEAM \
    --ppl-mbr \
    --remove-bpe \
    --batch-size 30 \
    > $OUTPUT_DIR/output.txt

cat $OUTPUT_DIR/output.txt \
    | grep ^H \
    | cut -f3- \
    | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' \
    > $OUTPUT_DIR/tmp/output.sys

cat $OUTPUT_DIR/output.txt \
    | grep ^T \
    | cut -f2- \
    | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' \
    > $OUTPUT_DIR/tmp/output.ref

perl /path-to-mosesdecoder/scripts/generic/multi-bleu.perl $OUTPUT_DIR/tmp/output.ref \
    < $OUTPUT_DIR/tmp/output.sys \
    > $OUTPUT_DIR/bleu.txt
    
echo "Finished $OUTPUT_NAME. BLEU:"
cat $OUTPUT_DIR/bleu.txt
