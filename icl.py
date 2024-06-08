import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
import json
import argparse
import random
import pickle
from tqdm import tqdm
import os
import logging

# Log config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    if len(a) == 0:
        return 0
    else:
        return round(sum(a) * 100 / float(len(a)), 2)

def icl_lm_eval_f1em(model, tokenizer, icl_examples, target, x):   
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

def icl_lm_eval_ppls(model, tokenizer, icl_examples, targets, x):
    device = torch.device(f'cuda:0')
    ppls = [] 
    for target in targets:
        tgt_len = len(tokenizer.encode(' ' + target))
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            ppl = torch.exp(outputs.loss)
            ppls.append(ppl.item())
    return ppls

def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning for pretrained GPTs")
    parser.add_argument("--lang1", type=str, default="")
    parser.add_argument("--lang2", type=str, default="")
    parser.add_argument("--indexdata", type=str, default="")
    parser.add_argument("--traindata", type=str, default="")
    parser.add_argument("--testdata", type=str, default="")
    parser.add_argument("--manualdata", type=str, default="")
    args = parser.parse_args()
    return args

device = 'cuda'
model_name = 'meta-llama/Meta-Llama-3-8B'

def construct_icl_examples(query_id, corpus_idx):
    icl_examples = []
    with open(f'./data/{args.traindata}{args.lang2}.json', 'r') as fIn: # mcounterfact_multi   zsre_multi   wfd_multi
        lines = json.load(fIn)
        demos = {entry["case_id"]: entry for entry in lines}
    if query_id in corpus_idx:
        # 获取对应的idxs
        demo_ids = corpus_idx[query_id]
        print(demo_ids)
        # 将每个index对应的example加入list
        for demo_id in demo_ids[:1]:
            if demo_id not in demos:
                print(f"Warning: demo_id {demo_id} 不在 demos 中，跳过此条目。")
                logging.warning(f"demo_id {demo_id} 不在 demos 中，跳过此条目。")
                continue
            line = demos[demo_id]
            new_fact = line['src']
            target_new = line['alt']
            prompt = line[args.lang2]['src']
            target_test = line[args.lang2]['alt']
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_test}\n\n')
    else:
        print("query_id not found")
    icl_examples.reverse()
    return icl_examples

def construct_icl_examples_manual(): 
    icl_examples = []
    with open(f'./data/manual_prompts/{args.manualdata}.json', 'r') as fIn: # mcounterfact_multi   zsre_multi   wfd_multi
        lines = json.load(fIn)
        for line in lines[:1]:
            lang1 = line['new_fact'] if args.lang1 == 'en' else args.lang1
            icl_examples.append(f"New Fact: {lang1} \nPrompt: {line[args.lang2]} \n\n")
    icl_examples.reverse()
    return icl_examples

def wrap_f1em_list(listf1, listem, ans, target):
    single_f1, single_em = obtain_f1_and_em(ans, target)
    listf1.append(single_f1)
    listem.append(single_em)

def wrap_ppls_count(edit_ppls, total_cnt, success_cnt, magnitude ):
    edit_final_probs = [1 / edit_ppls[0], 1 / edit_ppls[1]]
    total_cnt += 1
    if edit_final_probs[0] > edit_final_probs[1]:
        success_cnt += 1
    magnitude += edit_final_probs[0] - edit_final_probs[1]
    return total_cnt, success_cnt, magnitude

def read_corpus_idx(path):
    with open(f'./data/corpus_idx/{path}.json', 'r') as f:
        demos = json.load(f)
    demos_dict = {entry["query_id"]: entry["corpus_ids"] for entry in demos}
    return demos_dict

if __name__ == '__main__':
    device = torch.device(f'cuda:0')
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lines = []
    with open(f'./data/{args.testdata}{args.lang1}{args.lang2}.json', 'r') as f:
        lines = json.load(f)
    icl_examples = []
    success_cnt = 0
    para_success_cnt = 0
    magnitude = .0
    para_magnitude = .0
    orig_magnitude = .0
    total_cnt = 0
    para_total_cnt = 0
    orig_success_cnt = 0
    orig_total_cnt = 0
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

    corpus_idx = read_corpus_idx(args.indexdata) 

    example_idx = 0
    for i, line in enumerate(tqdm(lines[:5], total=len(lines[:30]), desc="Processing lines")):
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

        # print("#2")
        icl_examples = construct_icl_examples(i, corpus_idx)
        icl_examples_manual = construct_icl_examples_manual()
        icl_examples.append(f'New Fact: {prompts_truth} {target_truth}\nPrompt: {prompts_test}{target_test}\n\n')  # 要不要加prompts_test + target_test。  Prompt: {prompts_test}{target_test}\n\n
        print(f"icl_examples: {icl_examples[:10]}")
        print(f"icl_examples_manual : {icl_examples_manual[:10]}")
        print(f"prompts_truth: {prompts_truth}")
        print(f"prompts_test: {prompts_test}")
        print(f"target_test: {target_test}")

        if "MzsRE" in args.testdata:
            # reliablilty (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples, target_test, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {prompts_test}')
            wrap_f1em_list(reliablilty_f1_list, reliablilty_em_list, ans, target_test)
            print(f"reliablilty ans: {ans}, target_test: {target_test}")

            # generalization (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples_manual, target_test, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {rephrase_prompt}')
            wrap_f1em_list(generalization_f1_list, generalization_em_list, ans, target_test)
            print(f"generalization ans: {ans}, target_test: {target_test}")

            # locality (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples, locality_an, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {locality_prompt}' )
            wrap_f1em_list(locality_f1_list, locality_em_list, ans, locality_an)

            # portablility (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples_manual, portability_an, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {portability_prompt}'  )
            wrap_f1em_list(portablility_f1_list, portablility_em_list, ans, portability_an)
            print(f"portablility ans: {ans}, target_test: {portability_an}")

        elif  "MCounter" in args.testdata:
            # reliablilty (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples, [target_test, locality_an], f'New Fact: {prompts_truth} {target_truth}\nPrompt: {prompts_test}')
            orig_total_cnt, orig_success_cnt, orig_magnitude = wrap_ppls_count(edit_ppls, orig_total_cnt, orig_success_cnt, orig_magnitude)
            print(f"orig_total_cnt: {orig_total_cnt}")
            print(f"orig_success_cnt: {orig_success_cnt}")

            # generalization (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples, [target_test, locality_an], f'New Fact: {prompts_truth} {target_truth}\nPrompt: {rephrase_prompt}')
            para_total_cnt, para_success_cnt, para_magnitude = wrap_ppls_count(edit_ppls, para_total_cnt, para_success_cnt, para_magnitude)

            # locality (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples, [locality_an, target_test], f'New Fact: {prompts_truth} {target_truth}\nPrompt: {locality_prompt}')
            total_cnt, success_cnt, magnitude = wrap_ppls_count(edit_ppls, total_cnt, success_cnt, magnitude)
            print(f"success_cnt: {success_cnt}, total_cnt {total_cnt}")

            # portablility (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples_manual, portability_an, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {portability_prompt}')
            wrap_f1em_list(portablility_f1_list, portablility_em_list, ans, portability_an)
        elif  "WikiFact" in args.testdata:
            # reliablilty (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples, [target_test, locality_an], f'New Fact: {prompts_truth} {target_truth}\nPrompt: {prompts_test}')
            orig_total_cnt, orig_success_cnt, orig_magnitude = wrap_ppls_count(edit_ppls, orig_total_cnt, orig_success_cnt, orig_magnitude)

            # generalization (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples, [target_test, locality_an], f'New Fact: {prompts_truth} {target_truth}\nPrompt: {rephrase_prompt}')
            para_total_cnt, para_success_cnt, para_magnitude = wrap_ppls_count(edit_ppls, para_total_cnt, para_success_cnt, para_magnitude)

            # locality (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples, locality_an, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {locality_prompt}')
            wrap_f1em_list(locality_f1_list, locality_em_list, ans, locality_an)

            # portablility (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples_manual, portability_an, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {portability_prompt}')
            wrap_f1em_list(portablility_f1_list, portablility_em_list, ans, portability_an)
        else:
            print("unvalid test data")

        example_idx += 1
        # print(example_idx)

    # 打印结果
    print("F1 score")
    print("reliablilty_f1: %f" % (my_avg(reliablilty_f1_list)))
    print("generalization_f1: %f" % my_avg(generalization_f1_list))
    print("locality_f1: %f" % (my_avg(locality_f1_list)))
    print("portablility_f1: %f" % my_avg(portablility_f1_list))

    print("EM score")
    print("reliablilty_em: %f" % (my_avg(reliablilty_em_list)))
    print("generalization_em: %f" % my_avg(generalization_em_list))
    print("locality_em: %f"%my_avg(locality_em_list))
    print("portablility_em: %f" % my_avg(portablility_em_list))

    print("PPLS score")
    if orig_total_cnt != 0:
        print("reliablilty_ppls: %f, magnitude: %f" % (orig_success_cnt/orig_total_cnt*100, orig_magnitude/orig_total_cnt*100))
    if total_cnt != 0:
        print("locality_ppls: %f, magnitude: %f" % (success_cnt/total_cnt*100, magnitude/total_cnt*100))
    if para_total_cnt != 0:
        print("generalization_ppls: %f, magnitude: %f" % (para_success_cnt/para_total_cnt*100, para_magnitude/para_total_cnt*100))


    # 写入结果到文件
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_name = f'output_{args.testdata}_{args.lang1}{args.lang2}.txt'
    output_file_path = os.path.join(root_dir, output_file_name)
    output_folder = os.path.dirname(output_file_path)
    os.makedirs(output_folder, exist_ok=True)
    with open(output_file_path, 'w+') as f:
        f.write("F1 score\n")
        f.write("reliablilty_f1: %f\n" % (my_avg(reliablilty_f1_list)))
        f.write("generalization_f1: %f\n" % my_avg(generalization_f1_list))
        f.write("locality_f1: %f\n" % my_avg(locality_f1_list))
        f.write("portablility_f1: %f\n" % my_avg(portablility_f1_list))
        
        f.write("\nEM score\n")
        f.write("reliablilty_em: %f\n" % (my_avg(reliablilty_em_list)))
        f.write("generalization_em: %f\n" % my_avg(generalization_em_list))
        f.write("locality_em: %f\n" % my_avg(locality_em_list))
        f.write("portablility_em: %f\n" % my_avg(portablility_em_list))
        
        f.write("\nPPLS score\n")
        if orig_total_cnt != 0:
            f.write("reliablilty_ppls: %f, magnitude: %f\n" % (orig_success_cnt/orig_total_cnt*100, orig_magnitude/orig_total_cnt*100))
        if total_cnt != 0:
            f.write("locality_ppls: %f, magnitude: %f\n" % (success_cnt/total_cnt*100, magnitude/total_cnt*100))
        if para_total_cnt != 0:
            f.write("generalization_ppls: %f, magnitude: %f\n" % (para_success_cnt/para_total_cnt*100, para_magnitude/para_total_cnt*100))