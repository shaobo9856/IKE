#!/bin/bash

LANGS=("af" "ar" "az" "be" "bg" "bn" "ca" "ceb" "cs" "cy" "da" "de" "el" "es" "et" "eu" "fa" "fi" "fr" "ga" "gl" "he" "hi" "hr" "hu" "hy" "id" "it" "ja" "ka" "ko" "la" "lt" "lv" "ms" "nl" "pl" "pt" "ro" "ru" "sk" "sl" "sq" "sr" "sv" "ta" "th" "tr" "vi" "uk" "ur" "zh-cn")
CUDA=0

for LANG in "${LANGS[@]}";
do
    echo "currently processing languag: $LANG"
    CUDA_VISIBLE_DEVICES=$CUDA python icl.py --lang1 en --lang2 $LANG --testdata MzsRE/mzsre_test_duplicate_ --traindata MzsRE/zsre_mend_train_ --indexdata mzsre_corpus_idx --manualdata zsre_multi
done