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


def calculate_metrics(file_root):
    with open(file_root, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    for item in tqdm(data):
        reliablilty_f1, reliablilty_em = obtain_f1_and_em(item["post"]["reliability"]["ans"],
                                                          item["post"]["reliability"]["target"])
        reliablilty_f1_list.append(reliablilty_f1)
        reliablilty_em_list.append(reliablilty_em)

        generalization_f1, generalization_em = obtain_f1_and_em(item["post"]["generalization"]["rephrase_acc"]["ans"],
                                                                item["post"]["generalization"]["rephrase_acc"][
                                                                    "target"])
        generalization_f1_list.append(generalization_f1)
        generalization_em_list.append(generalization_em)

        locality_f1, locality_em = obtain_f1_and_em(item["post"]["locality"]["neighborhood_acc"]["ans"],
                                                          item["pre"]["locality"]["neighborhood_acc"]["ans"])
        locality_f1_list.append(locality_f1)
        locality_em_list.append(locality_em)


        portablility_f1, portablility_em = obtain_f1_and_em(item["post"]["portability"]["one_hop_acc"]["ans"],
                                                            item["post"]["portability"]["one_hop_acc"]["target"])
        portablility_f1_list.append(portablility_f1)
        portablility_em_list.append(portablility_em)



    print("=" * 20 + file_root + "=" * 20)
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

    reli, gene, loca, port = str(my_avg(reliablilty_f1_list)) + '/' + str(my_avg(reliablilty_em_list)),str(my_avg(generalization_f1_list)) + '/' + str(my_avg(generalization_em_list)),str(my_avg(locality_f1_list)) + '/' + str(my_avg(locality_em_list)),str(my_avg(portablility_f1_list)) + '/' + str(my_avg(portablility_em_list))

    return reli, gene, loca, port


def icl_lm_eval(
        model,
        tokenizer,
        icl_examples,
        prompt,
        neighborhood=False
):
    device = torch.device(f'cuda:0')
    # target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
    encodings = tokenizer(''.join(icl_examples) + f'{prompt}', return_tensors='pt',max_length=1520) 
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    ans = torch.argmax(logits, dim=-1) #[:,-target_ids.size(1):-1].squeeze()
    # ans_idss = ans.detach().cpu().numpy().tolist()
    # if not isinstance(ans_idss, list):
    #     ans_idss = [ans_idss]

    textual_ans = tokenizer.decode(ans[0], skip_special_tokens=True)

    if neighborhood:
        return textual_ans
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
        # if type_ == 0:
        #     icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {new_fact} {target_new}\n\n')
        # elif type_ == 1:
        #     prompt = random.choice(line['paraphrase_prompts'])
        #     icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_new}\n\n')
        # elif type_ == 2:
        #     prompt = random.choice(line['neighborhood_prompts'])
        #     icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_true}\n\n')
    icl_examples.reverse()
    return icl_examples


# def icl_lm_eval(model, tokenizer, icl_examples, targets, x):
#     ppls = [] 
#     for target in targets:
#         tgt_len = len(tokenizer.encode(' ' + target))
#         encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
#         input_ids = encodings['input_ids'].to(device)
#         target_ids = input_ids.clone()
#         target_ids[:, :-tgt_len] = -100
#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)
#             ppl = torch.exp(outputs.loss)
#             ppls.append(ppl.item())
#     return ppls

# def get_final_probs(yesno_ppls, icl_ppls, orig_ppls):
#     yes_prob = 1 / yesno_ppls[0]
#     no_prob = 1 / yesno_ppls[1]
#     final_probs = [yes_prob / icl_ppls[0] + no_prob / orig_ppls[0], yes_prob / icl_ppls[1] + no_prob / orig_ppls[1]]
#     return final_probs


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
        # relation = line['requested_rewrite']['relation_id']
        # prompt = line['requested_rewrite']['prompt']
        # subject = line['requested_rewrite']['subject']
        # prompt_calibrate = prompt.format('SUBJECT')
        # prompt = prompt.format(subject)
        # PROMPTS = [prompt, prompt_calibrate]
        en_data = line['en']
        id = en_data['id']
        prompt = en_data['prompt']
        new_fact = en_data['new_fact']
        type = en_data['type']
        print("#1")
        # target_true = line['requested_rewrite']['target_true']['str']
        # target_new = line['requested_rewrite']['target_new']['str']
        
        # PPLs = []
        # targets = [target_new, target_true]
        icl_examples = construct_icl_examples(example_idx, demos)       # 
        print("#2")

        icl_examples.append(f'New Fact: {new_fact}\nPrompt: {prompt}\n\n')
        ans = icl_lm_eval(model,tokenizer,icl_examples,prompt)
        print("#3")

        if type == "copy":
            reliablilty_f1, reliablilty_em = obtain_f1_and_em(ans, prompt[prompt.find('?')+1:])
            reliablilty_f1_list.append(reliablilty_f1)
            reliablilty_em_list.append(reliablilty_em)
        elif type == "update":
            generalization_f1, generalization_em = obtain_f1_and_em(ans, prompt[prompt.find('?')+1:])
            generalization_f1_list.append(generalization_f1)
            generalization_em_list.append(generalization_em)
        elif type == "retain":
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
