# This file is for debug only

# test dev
import torch
import os
import numpy as np

def test_teacher_logits_dist():
    #data_dict = {959: [], 839: []}
    data_dict = {0: [], 1: []}
    for epoch in range(300):
        print(epoch)
        data = torch.load('output/swin_tiny_patch4_window7_224/debug_da_trial_11/sample_target_rank_0_epoch_%d.pth'%epoch)
        target_index = torch.argmax(data['target'], axis=1)
        data_dict[target_index[0].item()].append(data['output'])
        data_dict[target_index[1].item()].append(data['output'])
        '''
        if target_index[0].item() in [979, 940]:
            data_dict[959].append(data['output'][0].tolist())
        else:
            data_dict[target_index[0].item()].append(data['output'][0].tolist())
        if target_index[1].item() in [940, 923]:
            data_dict[959].append(data['output'][1].tolist())
        else:
            data_dict[target_index[1].item()].append(data['output'][1].tolist())
        '''
    print(len(data_dict[0]), len(data_dict[1]))
    std0 = np.std(np.array(data_dict[0]), axis=0)
    print(std0, np.max(std0), np.min(std0), np.mean(std0))
    print('-'*20)
    std1 = np.std(np.array(data_dict[1]), axis=0)
    print(std1, np.max(std1), np.min(std1), np.mean(std1))



def test_da_random_seed():
    trial_list = [9, 10]
    data = {}
    for trial_id in trial_list:
        folder_name = 'output/swin_tiny_patch4_window7_224/debug_da_trial_%d'%trial_id
        data[trial_id] = {'sample': [], 'target': []}
        for rank in range(8):
            for iter in range(10):
                filename = os.path.join(folder_name, 'sample_target_rank_%d_iter_%d.pth'%(rank, iter))
                data_dict = torch.load(filename)
                data[trial_id]['sample'].append(data_dict['samples'])
                data[trial_id]['target'].append(data_dict['targets'])
    '''
    for target_0, target_1 in zip(data[0]['target'], data[1]['target']):
        print(target_0, target_1)
        print('-'*20)
    '''
    for sample_0, sample_1 in zip(data[trial_list[0]]['sample'], data[trial_list[1]]['sample']):
        print(torch.sum((sample_0-sample_1)**2))
        print('-'*20)

if __name__ == '__main__':
    #main()
    #test_da_random_seed()
    test_teacher_logits_dist()