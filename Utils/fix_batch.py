import os
import json
total_batch = 8

for idx in range(0,total_batch):
    command = f"nohup taskset -c {idx*4+2+32}-{idx*4+3+32} python fix_json.py --batch_num {idx} --total_batch {total_batch} > logs/fix_mask_json_{idx}.log 2>&1 &"
    print(command)
    os.system(command)