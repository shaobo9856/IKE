# BMIKE-53
Source code for "BMIKE-53: Investigating Cross-Lingual Knowledge Editing withIn-Context Learning"

## Overview

### Run command

cd search && python3 -m venv ike && source ike/bin/activate && pip install -r requirements.txt

pip install --upgrade sentence-transformers

python -m pip install huggingface_hub

huggingface-cli login --token 

python icl.py --lang1 en --lang2 af  --testdata MzsRE/mzsre_test_duplicate_   --traindata MzsRE/zsre_mend_train_   --indexdata mzsre_corpus_idx  --manualdata zsre_multi --lcount 10

python icl.py --lang1 en --lang2 af --testdata MCounterFact/mcounterfact_test_ --traindata MzsRE/zsre_mend_train_   --indexdata mzsre_corpus_idx  --manualdata zsre_multi --lcount 10

python icl.py --lang1 en --lang2 af --testdata WikiFactDiff/wfd_test_ --traindata MzsRE/zsre_mend_train_   --indexdata mzsre_corpus_idx  --manualdata zsre_multi    --lcount 10

chmod +x run_icls.sh
./run_icls.sh
