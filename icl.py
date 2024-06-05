import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import set_seed
# from transformers import GPT2Tokenizer, OPTForCausalLM
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
        prompt,
        target,
        neighborhood=False
):
    device = torch.device(f'cuda:0')
    # target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
    # encodings = tokenizer(''.join(icl_examples) + f'{prompt}', return_tensors='pt',max_length=1520) 
    # input_ids = encodings['input_ids'].to(device)
    # attention_mask = encodings['attention_mask'].to(device)
    # logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # ans = torch.argmax(logits, dim=-1) #[:,-target_ids.size(1):-1].squeeze()
    # # ans_idss = ans.detach().cpu().numpy().tolist()
    # # if not isinstance(ans_idss, list):
    # #     ans_idss = [ans_idss]

    # textual_ans = tokenizer.decode(ans[0], skip_special_tokens=True)

    # return textual_ans
    
    # 使用 generate 方法生成文本
    # with torch.no_grad():
    #     outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=50)
    
    # generated_ids = outputs[0][input_ids.shape[-1]:]  # 只保留生成的新token

    # # 解码生成的 token 为字符串
    # textual_ans = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # return textual_ans
    target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
    encodings = tokenizer(''.join(icl_examples) + f'{prompt}', return_tensors='pt',max_length=1520) # few shot  -> zero shot: ''.join(icl_examples) + 
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
    args = parser.parse_args()
    return args

device = 'cuda'
model_name = 'EleutherAI/gpt-j-6B'



with open('corpus_idx.json', 'r') as fIn:
    lines = json.load(fIn)
    
    for line in lines:
        print(line)
        print(line['corpus_ids'])

    corpus_idx = [ [int(idx) for idx in line['corpus_ids']] for line in lines]

def construct_icl_examples(idx, demos): # idx为前2000条的每一个index， demos为counterfact.json中2000条后
    # order = [2, 1, 2, 0, 1, 2, 2, 0, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2] #32 个元素。
    # random.shuffle(order)
    icl_examples = []
    demo_ids = corpus_idx[idx] # 获取对应idx的最相似的32条
    print(f"idx {idx}")
    print(f"demos: {demo_ids}")

    # demo_ids = demo_ids[:len(order)]
    for demo_id  in demo_ids:
        # print(f"demos: {demos}")
        print(demo_id)
        line = demos[demo_id-11]
        print(line)
        new_fact = en_data['new_fact']
        prompt = en_data['prompt']
        type_ = en_data['type']

        icl_examples.append(f'New Fact: {new_fact} \nPrompt: {prompt} \n\n')
    icl_examples.reverse()
    return icl_examples


if __name__ == '__main__':
    # random.seed(42)
    args = parse_args()
    seed = args.seed
    set_seed(seed)
    model = GPTJForCausalLM.from_pretrained(model_name).to(device)
    # model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    # model = GPTNeoXForCausalLM.from_pretrained(model_name).half().to(device)
    # model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    # tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)
    # model = OPTForCausalLM.from_pretrained("facebook/opt-13b").to(device)
    # tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-13b")

    lines = []

    with open('./data/zsre.json', 'r') as f:
        lines = json.load(f)
    icl_examples = []
    demos = lines[10:]
    lines = lines[:10]
    calibrate_magnitude = .0
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

    # icl_cnt = 0
    example_idx = 0
    for i, line in enumerate(lines): # 前10条

        if i % 10 == 0:
            print(i, success_cnt, total_cnt, magnitude / (total_cnt + 1e-12), para_success_cnt, para_magnitude / (para_total_cnt + 1e-12), orig_success_cnt ,orig_magnitude / (i + 1e-12))
        en_data = line['en']
        id = en_data['id']
        prompt = en_data['prompt']
        new_fact = en_data['new_fact']
        type = en_data['type']
        print("#1")


        icl_examples = construct_icl_examples(example_idx, demos)       # 
        print("#2")

        icl_examples.append(f'New Fact: {new_fact}\nPrompt: {prompt}\n\n') # ? 
        ans = icl_lm_eval(model,tokenizer,icl_examples,prompt,prompt[prompt.find('?')+1:])
        print("#3")
        print(f"ans:{ans}, target: {prompt[prompt.find('?')+1:]}")
        if type == "copy":
            reliablilty_f1, reliablilty_em = obtain_f1_and_em(ans, prompt[prompt.find('?')+1:])
            reliablilty_f1_list.append(reliablilty_f1)
            reliablilty_em_list.append(reliablilty_em)
        elif type == "update":
            generalization_f1, generalization_em = obtain_f1_and_em(ans, prompt[prompt.find('?')+1:])
            generalization_f1_list.append(generalization_f1)
            generalization_em_list.append(generalization_em)
        elif type == "retain":
            locality_f1, locality_em = obtain_f1_and_em(ans, prompt[prompt.find('?')+1:])
            locality_f1_list.append(locality_f1)
            locality_em_list.append(locality_em)
        elif type == "portability":
            portablility_f1, portablility_em =  obtain_f1_and_em(ans, prompt[prompt.find('?')+1:])
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

    # ppls: 一个包含困惑度（Perplexity, PPL）的列表。每个目标文本对应一个困惑度值。 icl_examples: 构建的 ICL 示例列表，这些示例将作为模型的输入上下文。 targets: 目标字符串列表，即模型需要预测的目标文本。x: 追加到 ICL 示例后的查询文本（query text）。
    #     edit_ppls = icl_lm_eval(model, tokenizer, icl_examples, [target_new, target_true], f'New Fact: {prompt} {target_new}\nPrompt: {prompt}')

    #     edit_final_probs = [1 / edit_ppls[0], 1 / edit_ppls[1]]         # 如果 edit_ppls[0]（target_new 的困惑度）明显低于 edit_ppls[1]（target_true 的困惑度），说明模型更倾向于接受新事实。
    #     orig_total_cnt += 1
    #     if edit_final_probs[0] > edit_final_probs[1]:
    #         orig_success_cnt += 1
    #     orig_magnitude += edit_final_probs[0] - edit_final_probs[1]


    #     targets = [target_new, target_true]

    #     paraphrases = line['paraphrase_prompts']
    #     for paraphrase in paraphrases:
    #         paraphrase_ppls = icl_lm_eval(model, tokenizer, icl_examples, [target_new, target_true], f'New Fact: {prompt} {target_new}\nPrompt: {paraphrase}')
    #         paraphrase_final_probs = [1 / paraphrase_ppls[0], 1 / paraphrase_ppls[1]]
            
    #         if paraphrase_final_probs[0] > paraphrase_final_probs[1]:
    #             para_success_cnt += 1
    #         para_magnitude += paraphrase_final_probs[0] - paraphrase_final_probs[1]
    #         para_total_cnt += 1

    #     neighbors = line['neighborhood_prompts']
    #     for neighbor in neighbors:
    #         neighbor_ppls = icl_lm_eval(model, tokenizer, icl_examples, [target_true, target_new], f'New Fact: {prompt} {target_new}\nPrompt: {neighbor}')
    #         neighbor_final_probs = [1 / neighbor_ppls[0], 1 / neighbor_ppls[1]]
            
    #         if neighbor_final_probs[0] > neighbor_final_probs[1]:
    #             success_cnt += 1
    #         magnitude += neighbor_final_probs[0] - neighbor_final_probs[1]
    #         total_cnt += 1



    # print(success_cnt/total_cnt, magnitude/total_cnt, para_success_cnt/para_total_cnt, para_magnitude/para_total_cnt, orig_success_cnt/orig_total_cnt, orig_magnitude/orig_total_cnt)
