#Note: unfinished!

import csv
import json
import os

tmp_dir = 'tmp'
output_path = 'results.csv'

with open(output_path, 'wt') as f_writer:
    writer = csv.writer(f_writer)

    for file in os.listdir(tmp_dir):
        with open(os.path.join(tmp_dir, file), 'rt') as f:
            metrics = json.load(f)
