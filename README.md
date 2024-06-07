# IKE
Source code for "Can We Edit Factual Knowledge by In-Context Learning?"
## Overview
## Requirements
```
jsonlines==3.1.0
nltk==3.6.7
numpy==1.22.3
openai==0.25.0
sentence_transformers==2.2.0
spacy==3.2.3
torch==1.11.0
tqdm==4.56.0
transformers==4.24.0

```
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

### Contrastive Knowledge Assessment

git clone https://github.com/shaobo9856/IKE ike

cd ike && git checkout manual && python3 -m venv ike && source ike/bin/activate && pip install -r requirements.txt

pip install --upgrade sentence-transformers

python -m pip install huggingface_hub

huggingface-cli login --token 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

python icltest.py --lang1 en --lang2 af --tdata mzsre_test_duplicate_ --pdata zsre_multi 

python icl.py --lang1 en --lang2 af --tdata MzsRE/mzsre_test_duplicate_ --pdata zsre_multi 
python icl.py --lang1 en --lang2 af --tdata MCounterFact/mcounterfact_test_ --pdata zsre_multi  
python icl.py --lang1 en --lang2 af --tdata WikiFactDiff/wfd_test_ --pdata zsre_multi  

chmod +x run_icls.sh
./run_icls.sh


encode_facts.py：  这段代码的主要目的是读取 counterfact.json 文件中的数据，生成一系列句子并使用 SentenceTransformer 模型将这些句子转换为句子嵌入，然后将这些嵌入存储到一个 pickle 文件中以便后续使用。

semantic_search.py： 将先前生成的句子嵌入分成查询集和语料库集，并使用 SentenceTransformer 进行语义搜索。它输出每个查询的最相似的句子的 ID。 保存到corpus_idx.txt文件











