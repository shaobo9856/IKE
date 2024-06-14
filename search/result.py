import argparse
import os
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="zsre", choices=['zsre', 'mcounterfact', 'wfd'])
parser.add_argument("--type", type=str, default="search", choices=['manual', 'search'])

args = parser.parse_args()

parent_path = f"IKE-{args.type}"

if args.task == 'zsre':
    subdir_path = "output_MzsRE"
elif args.task == 'mcounterfact':
    subdir_path = 'output_MCounterFact'
elif args.task == 'wfd':
    subdir_path = 'output_WikiFactDiff'

metrics = dict()
dir_path = os.path.join(parent_path, subdir_path)
for file_name in os.listdir(dir_path):
    idx_start = file_name.index('en') + 2
    lang = file_name[idx_start: idx_start + 2]
    metric_lang = defaultdict(dict)
    
    with open(os.path.join(dir_path, file_name), 'rt') as f:
        lines = f.readlines()
        if args.task == 'zsre' and args.type == 'manual':
            metric_lang['reliability']['f1'] = lines[1].strip().split(':')[-1].strip()
            metric_lang['generality']['f1'] = lines[2].strip().split(':')[-1].strip()
            metric_lang['locality']['f1'] = lines[3].strip().split(':')[-1].strip()
            metric_lang['portability']['f1'] = lines[4].strip().split(':')[-1].strip()
            metric_lang['reliability']['em'] = lines[7].strip().split(':')[-1].strip()
            metric_lang['generality']['em'] = lines[8].strip().split(':')[-1].strip()
            metric_lang['locality']['em'] = lines[9].strip().split(':')[-1].strip()
            metric_lang['portability']['em'] = lines[10].strip().split(':')[-1].strip()
            
        elif args.task == 'zsre' and args.type == 'search':
            metric_lang['reliability']['f1'] = lines[1].strip().split('   ')[0].split(':')[-1].strip()
            metric_lang['reliability']['em'] = lines[1].strip().split('   ')[1].split(':')[-1].strip()
            metric_lang['generality']['f1'] = lines[2].strip().split('   ')[0].split(':')[-1].strip()
            metric_lang['generality']['em'] = lines[2].strip().split('   ')[1].split(':')[-1].strip()
            metric_lang['locality']['f1'] = lines[3].strip().split('   ')[0].split(':')[-1].strip()
            metric_lang['locality']['em'] = lines[3].strip().split('   ')[1].split(':')[-1].strip()
            metric_lang['portability']['f1'] = lines[4].strip().split('   ')[0].split(':')[-1].strip()
            metric_lang['portability']['em'] = lines[4].strip().split('   ')[1].split(':')[-1].strip()
        
        elif args.task == 'mcounterfact' and args.type == 'manual':
            metric_lang['reliability']['ppls'] = lines[13].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['reliability']['magnitude'] = lines[13].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['generalization']['ppls'] = lines[15].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['generalization']['magnitude'] = lines[15].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['locality']['ppls'] = lines[14].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['locality']['magnitude'] = lines[14].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['portability']['f1'] = lines[4].strip().split(':')[-1].strip()
            metric_lang['portability']['em'] = lines[10].strip().split(':')[-1].strip()
        
        elif args.task == 'mcounterfact' and args.type == 'search':
            metric_lang['reliability']['ppls'] = lines[4].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['reliability']['magnitude'] = lines[4].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['generalization']['ppls'] = lines[6].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['generalization']['magnitude'] = lines[6].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['locality']['ppls'] = lines[5].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['locality']['magnitude'] = lines[5].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['portability']['f1'] = lines[1].strip().split('   ')[0].split(':')[-1].strip()
            metric_lang['portability']['em'] = lines[1].strip().split('   ')[1].split(':')[-1].strip()
            
        elif args.task == 'wfd' and args.type == 'manual':
            metric_lang['reliability']['ppls'] = lines[13].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['reliability']['magnitude'] = lines[13].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['generalization']['ppls'] = lines[14].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['generalization']['magnitude'] = lines[14].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['locality']['f1'] = lines[3].strip().split(':')[-1].strip()
            metric_lang['locality']['em'] = lines[9].strip().split(':')[-1].strip()
            metric_lang['portability']['f1'] = lines[4].strip().split(':')[-1].strip()
            metric_lang['portability']['em'] = lines[10].strip().split(':')[-1].strip()
            
        elif args.task == 'wfd' and args.type == 'search':
            metric_lang['reliability']['ppls'] = lines[5].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['reliability']['magnitude'] = lines[5].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['generalization']['ppls'] = lines[6].strip().split(', ')[0].split(':')[-1].strip()
            metric_lang['generalization']['magnitude'] = lines[6].strip().split(', ')[1].split(':')[-1].strip()
            metric_lang['locality']['f1'] = lines[1].strip().split('   ')[0].split(':')[-1].strip()
            metric_lang['locality']['em'] = lines[1].strip().split('   ')[1].split(':')[-1].strip()
            metric_lang['portability']['f1'] = lines[2].strip().split('   ')[0].split(':')[-1].strip()
            metric_lang['portability']['em'] = lines[2].strip().split('   ')[1].split(':')[-1].strip()
    
    metrics[lang] = metric_lang


# print(metrics)
save_path = f"tmp/{args.task}_{args.type}_tmp.json"
with open(save_path, 'wt') as f_json:
    json.dump(metrics, f_json, indent=2)