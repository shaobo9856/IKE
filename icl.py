import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
import json
import argparse
import random
import pickle
from tqdm import tqdm
import os


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
    parser.add_argument("--manualdata", type=str, default="")
    parser.add_argument("--testdata", type=str, default="")
    parser.add_argument("--lcount", type=int, default=3000)
    parser.add_argument('--pbase', action='store_true', help='Prompt baseline')
    args = parser.parse_args()
    return args

device = 'cuda'
model_name = 'meta-llama/Meta-Llama-3-8B'

def construct_icl_examples(): 
    icl_examples = []
    with open(f'./data/manual_prompts/{args.manualdata}.json', 'r') as fIn: # mcounterfact_multi   zsre_multi   wfd_multi
        lines = json.load(fIn)
        update_lines = [line for line in lines if line["type"] == "update"]

        for line in lines[:4]:
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

    # Print Switch
    f1r,f1g,f1l,f1p,pplr,pplg,ppll = False,False,False,False,False,False,False

    example_idx = 0
    if not args.pbase:
        icl_examples = construct_icl_examples()
    for i, line in enumerate(tqdm(lines[:args.lcount], total=len(lines[:args.lcount]), desc="Processing lines")):
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

        # icl_examples.append(f'New Fact: {prompts_truth} {target_truth}\nPrompt: {prompts_test}{target_test}\n\n')  # 要不要加prompts_test + target_test。  Prompt: {prompts_test}{target_test}\n\n

        if "MzsRE" in args.testdata:
            # reliablilty (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples, target_test, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {prompts_test}')
            wrap_f1em_list(reliablilty_f1_list, reliablilty_em_list, ans, target_test)

            # generalization (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples, target_test, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {rephrase_prompt}')
            wrap_f1em_list(generalization_f1_list, generalization_em_list, ans, target_test)

            # locality (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples, locality_an, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {locality_prompt}' )
            wrap_f1em_list(locality_f1_list, locality_em_list, ans, locality_an)

            # portablility (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples, portability_an, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {portability_prompt}'  )
            wrap_f1em_list(portablility_f1_list, portablility_em_list, ans, portability_an)

            f1r,f1g,f1l,f1p = True,True,True,True
        elif  "MCounter" in args.testdata:
            # reliablilty (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples, [target_test, locality_an], f'New Fact: {prompts_truth} {target_truth}\nPrompt: {prompts_test}')
            orig_total_cnt, orig_success_cnt, orig_magnitude = wrap_ppls_count(edit_ppls, orig_total_cnt, orig_success_cnt, orig_magnitude)

            # generalization (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples, [target_test, locality_an], f'New Fact: {prompts_truth} {target_truth}\nPrompt: {rephrase_prompt}')
            para_total_cnt, para_success_cnt, para_magnitude = wrap_ppls_count(edit_ppls, para_total_cnt, para_success_cnt, para_magnitude)

            # locality (ppls)
            edit_ppls = icl_lm_eval_ppls(model,tokenizer, icl_examples, [locality_an, target_test], f'New Fact: {prompts_truth} {target_truth}\nPrompt: {locality_prompt}')
            total_cnt, success_cnt, magnitude = wrap_ppls_count(edit_ppls, total_cnt, success_cnt, magnitude)
            # print(f"success_cnt: {success_cnt}, total_cnt {total_cnt}")

            # portablility (f1em)
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples, portability_an, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {portability_prompt}')
            wrap_f1em_list(portablility_f1_list, portablility_em_list, ans, portability_an)

            f1p,pplr,pplg,ppll = True,True,True,True
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
            ans = icl_lm_eval_f1em(model,tokenizer, icl_examples, portability_an, f'New Fact: {prompts_truth} {target_truth}\nPrompt: {portability_prompt}')
            wrap_f1em_list(portablility_f1_list, portablility_em_list, ans, portability_an)

            f1l,f1p,pplr,pplg = True,True,True,True
        else:
            print("unvalid test data")

        example_idx += 1
        # print(example_idx)

    # 打印结果
    print("F1EM score")
    if f1r: print("reliablilty_f1: %f   reliablilty_em: %f" % (my_avg(reliablilty_f1_list), my_avg(reliablilty_em_list)))
    if f1g: print("generalization_f1: %f    generalization_em: %f" % (my_avg(generalization_f1_list), my_avg(generalization_em_list)))
    if f1l: print("locality_f1: %f  locality_em: %f" % (my_avg(locality_f1_list), my_avg(locality_em_list)))
    if f1p: print("portablility_f1: %f  portablility_em: %f" % (my_avg(portablility_f1_list), my_avg(portablility_em_list)))

    print("PPLS score")
    if pplr: print("reliablilty_ppls: %f, magnitude: %f" % (orig_success_cnt/orig_total_cnt*100, orig_magnitude/orig_total_cnt*100))
    if ppll: print("locality_ppls: %f, magnitude: %f" % (success_cnt/total_cnt*100, magnitude/total_cnt*100))
    if pplg: print("generalization_ppls: %f, magnitude: %f" % (para_success_cnt/para_total_cnt*100, para_magnitude/para_total_cnt*100))


    # 写入结果到文件
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_name = f'output_{args.testdata}_{args.lang1}{args.lang2}.txt'
    output_file_path = os.path.join(root_dir, output_file_name)
    output_folder = os.path.dirname(output_file_path)
    os.makedirs(output_folder, exist_ok=True)

    with open(output_file_path, 'w+') as f:
        f.write("F1EM score\n")
        if f1r:
            f.write(f"reliability_f1: {my_avg(reliablilty_f1_list):.6f}   reliability_em: {my_avg(reliablilty_em_list):.6f}\n")
        if f1g:
            f.write(f"generalization_f1: {my_avg(generalization_f1_list):.6f}   generalization_em: {my_avg(generalization_em_list):.6f}\n")
        if f1l:
            f.write(f"locality_f1: {my_avg(locality_f1_list):.6f}   locality_em: {my_avg(locality_em_list):.6f}\n")
        if f1p:
            f.write(f"portability_f1: {my_avg(portablility_f1_list):.6f}   portability_em: {my_avg(portablility_em_list):.6f}\n")
        
        f.write("\nPPLS score\n")
        if pplr and orig_total_cnt != 0:
            f.write(f"reliability_ppls: {orig_success_cnt/orig_total_cnt*100:.6f}, magnitude: {orig_magnitude/orig_total_cnt*100:.6f}\n")
        if ppll and total_cnt != 0:
            f.write(f"locality_ppls: {success_cnt/total_cnt*100:.6f}, magnitude: {magnitude/total_cnt*100:.6f}\n")
        if pplg and para_total_cnt != 0:
            f.write(f"generalization_ppls: {para_success_cnt/para_total_cnt*100:.6f}, magnitude: {para_magnitude/para_total_cnt*100:.6f}\n")