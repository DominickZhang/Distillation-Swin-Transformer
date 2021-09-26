# This file is for debug only

# test dev
import torch
import os

def main():
    trial_list = [0, 1]
    data = {}
    for trial_id in trial_list:
        folder_name = '/output/swin_tiny_patch4_window7_224/debug_da_trial_%d'%trial_id
        data[trial_id] = {'sample': [], 'target': []}
        for rank in range(8):
            for iter in range(10):
                filename = os.path.join(folder_name, 'sample_target_rank_%d_iter_%d.pth'%(rank, iter))
                data_dict = torch.load(filename)
                data[trial_id]['sample'].append(data_dict['samples'])
                data[trial_id]['target'].append(data_dict['targets'])
    for target_0, target_1 in zip(data[0]['target'], data[1]['target']):
        print(target_0, target_1)
        print('-'*20)

if __name__ == '__main__':
    main()