#!/bin/bash

LANGS=("es" "vi" "ru" "zh-cn") # "fr" "nl" "hu" "tr" "ca" "ja" 
CUDA=1

# run zsre
for LANG in "${LANGS[@]}";
do
    echo "currently processing languag: $LANG on zsRE"
    CUDA_VISIBLE_DEVICES=$CUDA python icl.py --lang1 en --lang2 $LANG --testdata MzsRE/mzsre_test_duplicate_ --manualdata zsre_multi  --lcount 10000
done

# # run mcounterfact
# for LANG in "${LANGS[@]}";
# do
#     echo "currently processing languag: $LANG on MCounterFact"
#     CUDA_VISIBLE_DEVICES=$CUDA python icl.py --lang1 en --lang2 $LANG  --testdata MCounterFact/mcounterfact_test_ --manualdata mcounterfact_multi  --lcount 10000
# done

# # run mcounterfact
# for LANG in "${LANGS[@]}";
# do
#     echo "currently processing languag: $LANG on WikiFactDiff"
#     CUDA_VISIBLE_DEVICES=$CUDA python icl.py --lang1 en --lang2 $LANG  --testdata WikiFactDiff/wfd_test_ --manualdata wfd_multi  --lcount 10000
# done