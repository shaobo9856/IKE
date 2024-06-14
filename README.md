# IKE
Source code for "BMIKE-53: Investigating Cross-Lingual Knowledge Editing withIn-Context Learning"

## Overview

### Contrastive Knowledge Assessment


cd ike && git checkout search && python3 -m venv ike && source ike/bin/activate && pip install -r requirements.txt

pip install --upgrade sentence-transformers

python -m pip install huggingface_hub

huggingface-cli login --token 

python icl.py --lang1 en --lang2 af  --testdata MzsRE/mzsre_test_duplicate_   --traindata MzsRE/zsre_mend_train_   --indexdata mzsre_corpus_idx  --manualdata zsre_multi --lcount 10

python icl.py --lang1 en --lang2 af --testdata MCounterFact/mcounterfact_test_ --traindata MzsRE/zsre_mend_train_   --indexdata mzsre_corpus_idx  --manualdata zsre_multi --lcount 10

python icl.py --lang1 en --lang2 af --testdata WikiFactDiff/wfd_test_ --traindata MzsRE/zsre_mend_train_   --indexdata mzsre_corpus_idx  --manualdata zsre_multi    --lcount 10

chmod +x run_icls.sh
./run_icls.sh



## How to run our experiments?
### Data Preparation
We conduct experiments on `CounterFact` dataset. You can download the dataset [here](https://rome.baulab.info/data/dsets/counterfact.json).
run code `clean_paraphrase.py` to remove unrelated prefixes in paraphrase prompts in `CounterFact` and keep all prompts in the same format.

### Demonstration Organization
We select first 2,000 records as test set. For each record, we use sentence transformers to retrieve 32 nearest neighbours from remaining records as in-context demonstrations. The indices of nearest neighbours are stored in `corpus_idx.txt`. You can also run code `encode_facts.py` and `semantic_search.py` to build `corpus_idx.txt`. 

### Main Experiments
IKE: `python icl.py`

PROMPT: `python prompt.py`

Other baselines: implemented by [rome](https://github.com/kmeng01/rome)

### Models for IKE

You can evaluate IKE based on different LLMs by specifying the model name of LLMs.

```
python icl.py --model_name [model name]
```
The model name can be `['gpt2-xl', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']`