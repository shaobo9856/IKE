# BMIKE-53
Source code for "BMIKE-53: Investigating Cross-Lingual Knowledge Editing withIn-Context Learning"

## Overview

### Run command

cd manual && python3 -m venv ike && source ike/bin/activate && pip install -r requirements.txt

pip install --upgrade sentence-transformers

python -m pip install huggingface_hub

huggingface-cli login --token 

python icltest.py --lang1 en --lang2 af --tdata mzsre_test_duplicate_ --pdata zsre_multi 

python icl.py --lang1 en --lang2 af --testdata MzsRE/mzsre_test_duplicate_ --manualdata zsre_multi --lcount 10
python icl.py --lang1 en --lang2 af --testdata MCounterFact/mcounterfact_test_ --manualdata zsre_multi  --lcount 10
python icl.py --lang1 en --lang2 af --testdata WikiFactDiff/wfd_test_ --manualdata zsre_multi  --lcount 10

chmod +x run_manual_ab.sh
./run_manual_ab.sh


CUDA_VISIBLE_DEVICES=0  python icl.py --lang1 en --lang2 af --testdata MzsRE/mzsre_test_duplicate_ --manualdata zsre_multi --lcount 10


