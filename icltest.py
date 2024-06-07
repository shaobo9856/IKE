import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed

import json
import argparse
import random
import pickle
from tqdm import tqdm


def obtain_f1_and_em(a, b):
    global tokenizer
    a_words = tokenizer.encode(a, add_special_tokens=False)
    b_words = tokenizer.encode(b, add_special_tokens=False)
    if len(a_words) == 0 and len(b_words) == 0:
        return 1.0, 1
    if len(a_words) == 0 or len(b_words) == 0:
        return 0.0, 0

    em = 1 if a == b else 0
    k = len(a_words) * len(b_words)

    intersecting_words = []
    for word in a_words.copy():
        if word in b_words:
            a_words.remove(word)
            b_words.remove(word)
            intersecting_words.append(word)

    f1_score = (len(intersecting_words) * len(intersecting_words)) / float(k)
    return f1_score, em


def my_avg(a):
    return round(sum(a) * 100 / float(len(a)), 2)


def icl_lm_eval(
        model,
        tokenizer,
        icl_examples,
        target,
        x,
):   
    device = torch.device(f'cuda:0')
    target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
    encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt',max_length=1520) # few shot  -> zero shot: ''.join(icl_examples) + 
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
    target_ids = target_ids[:,1:]
    
    ans_idss = ans.detach().cpu().numpy().tolist()
    target_idss = target_ids.detach().cpu().squeeze().numpy().tolist()
    if not isinstance(ans_idss, list):
        ans_idss = [ans_idss]

    textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True)
    return textual_ans

def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning for pretrained GPTs")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--lang1", type=str, default="")
    parser.add_argument("--lang2", type=str, default="")
    parser.add_argument("--pdata", type=str, default="")
    parser.add_argument("--tdata", type=str, default="")
    parser.add_argument("--metrics", type=str, default="")
    args = parser.parse_args()
    return args

device = 'cuda'
model_name = 'meta-llama/Meta-Llama-3-8B'


def construct_icl_examples(): 
    icl_examples = []
    with open(f'./data/manual_prompts/{args.pdata}.json', 'r') as fIn: # mcounterfact_multi   zsre_multi   wfd_multi
        lines = json.load(fIn)
        for line in lines:
            print(line['new_fact'])
            lang1 = line['new_fact'] if args.lang1 == 'en' else args.lang1
            icl_examples.append(f"New Fact: {lang1} \nPrompt: {line[args.lang2]} \n\n")
    icl_examples.reverse()
    return icl_examples


if __name__ == '__main__':
    # random.seed(42)
    args = parse_args()
    seed = args.seed
    set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lines = []
    with open(f'./data/MzsRE/{args.tdata}{args.lang1}{args.lang2}.json', 'r') as f:
        lines = json.load(f)
    icl_examples = []
    # calibrate_magnitude = .0
    # success_cnt = 0
    # para_success_cnt = 0
    # magnitude = .0
    # para_magnitude = .0
    # orig_magnitude = .0
    # total_cnt = 0
    # para_total_cnt = 0
    # orig_success_cnt = 0
    # orig_total_cnt = 0
    reliablilty_f1_list = []
    reliablilty_em_list = []
    generalization_f1_list = []
    generalization_em_list = []

    locality_f1_list = []
    locality_em_list = []
    specificity_f1_list = []
    specificity_em_list = []

    portablility_f1_list = []
    portablility_em_list = []

    # icl_cnt = 0
    example_idx = 0
    icl_examples = construct_icl_examples()
    for i, line in enumerate(lines): 

        # if i % 10 == 0:
        #     print(i, success_cnt, total_cnt, magnitude / (total_cnt + 1e-12), para_success_cnt, para_magnitude / (para_total_cnt + 1e-12), orig_success_cnt ,orig_magnitude / (i + 1e-12))
        subject = line[args.lang1]['subject']
        prompts_truth = line[args.lang1]['src']
        prompts_test = line[args.lang2]['src']

        target_truth = line[args.lang1]['alt']
        target_test = line[args.lang2]['alt']

        rephrase_prompt = line[args.lang2]['rephrase']
        locality_prompt = line[args.lang2]['loc']
        locality_an = line[args.lang2]['loc_ans']
        portability_prompt = line[args.lang2]['portability']['New Question']
        portability_an = line[args.lang2]['portability']['New Answer'] 

        print("#2")

        # icl_examples.append(f'New Fact: {prompts_truth} {target_truth}\nPrompt: {prompts_test}{target_test}\n\n')  # 要不要加prompts_test + target_test。  Prompt: {prompts_test}{target_test}\n\n

        # reliablilty
        ans = icl_lm_eval(model,tokenizer, icl_examples, target_test, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {prompts_test}')
        print(f"ans:{ans}, target: {target_test}")
        reliablilty_f1, reliablilty_em = obtain_f1_and_em(ans, target_test)
        reliablilty_f1_list.append(reliablilty_f1)
        reliablilty_em_list.append(reliablilty_em)

        # generalization
        ans = icl_lm_eval(model,tokenizer, icl_examples, target_test, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {rephrase_prompt}')
        generalization_f1, generalization_em = obtain_f1_and_em(ans, target_test)
        generalization_f1_list.append(generalization_f1)
        generalization_em_list.append(generalization_em)

        # locality
        ans = icl_lm_eval(model,tokenizer, icl_examples, locality_an, locality_prompt)
        locality_f1, locality_em = obtain_f1_and_em(ans, locality_an)
        locality_f1_list.append(locality_f1)
        locality_em_list.append(locality_em)

        # portablility
        ans = icl_lm_eval(model,tokenizer, icl_examples, portability_an, portability_prompt)
        portablility_f1, portablility_em =  obtain_f1_and_em(ans, portability_an)
        portablility_f1_list.append(portablility_f1)
        portablility_em_list.append(portablility_em)

        example_idx += 1
        print(example_idx)

    print("F1 score")
    print("reliablilty_f1: %f" % (my_avg(reliablilty_f1_list)))
    print("generalization_f1: %f" % my_avg(generalization_f1_list))
    print("locality_f1: %f"%my_avg(locality_f1_list))
    print("portablility_f1: %f" % my_avg(portablility_f1_list))

    print("EM score")
    print("reliablilty_em: %f" % (my_avg(reliablilty_em_list)))
    print("generalization_em: %f" % my_avg(generalization_em_list))
    print("locality_em: %f"%my_avg(locality_em_list))
    print("portablility_em: %f" % my_avg(portablility_em_list))