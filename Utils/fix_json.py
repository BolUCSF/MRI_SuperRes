import json,os
import nibabel as nib
import argparse
import numpy as np


with open('json/high_res.json') as f:
    images_files = json.load(f)

def main(args):
    batch_num = args.batch_num
    total_num = len(images_files)
    total_batch = args.total_batch
    batch_size = total_num // total_batch
    file_list = images_files[batch_size*batch_num:min(batch_size*(batch_num+1),total_num)]
    pair_list = []
    count = 0
    for file in file_list:
        tnum = file['file_name'].split('/')[-1].replace('_bet.nii.gz','')
        fl_path = file['file_name'].replace('t1c_bet','fl_bet')
        fl_mask_path = fl_path.replace('bet.nii.gz','bet_mask.nii.gz')
        if os.path.exists(fl_path):
            fl_nib = nib.load(fl_mask_path)
            fl_data = fl_nib.get_fdata()
            # print(np.sum(fl_data))
            pair_list.append(np.sum(fl_data))
            if np.sum(fl_data) < 320000:
                print(tnum)
    print(np.min(pair_list))
            
    with open(f'logs/mask_val_check_{batch_num}.json','w') as f:
        json.dump(pair_list,f,indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_num', type=int)
    parser.add_argument('--total_batch', type=int)
    parser.add_argument('--type', type=str)
    args = parser.parse_args()
    main(args)
