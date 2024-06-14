#!/bin/bash

LANGS=("af" "ar" "az" "be" "bg" "bn" "ca" "ceb" "cs" "cy" "da" "de" "el" "es" "et" "eu" "fa" "fi" "fr" "ga" "gl" "he" "hi" "hr" "hu" "hy" "id" "it" "ja" "ka" "ko" "la" "lt" "lv" "ms" "nl" "pl" "pt" "ro" "ru" "sk" "sl" "sq" "sr" "sv" "ta" "th" "tr" "vi" "uk" "ur" "zh-cn")
CUDA=3

# run zsre
for LANG in "${LANGS[@]}";
do
    echo "currently processing languag: $LANG on zsRE"
    CUDA_VISIBLE_DEVICES=$CUDA python icl.py --lang1 en --lang2 $LANG --testdata MzsRE/mzsre_test_duplicate_ --manualdata zsre_multi  --lcount 10000
done

# run mcounterfact
for LANG in "${LANGS[@]}";
do
    echo "currently processing languag: $LANG on MCounterFact"
    CUDA_VISIBLE_DEVICES=$CUDA python icl.py --lang1 en --lang2 $LANG  --testdata MCounterFact/mcounterfact_test_ --manualdata mcounterfact_multi  --lcount 10000
done

# run mcounterfact
for LANG in "${LANGS[@]}";
do
    echo "currently processing languag: $LANG on WikiFactDiff"
    CUDA_VISIBLE_DEVICES=$CUDA python icl.py --lang1 en --lang2 $LANG  --testdata WikiFactDiff/wfd_test_ --manualdata wfd_multi  --lcount 10000
done