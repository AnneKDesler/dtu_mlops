
import argparse
import time

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from lfw_dataset import LFWDataset

      
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
            transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((20,20)),
            transforms.ToTensor()
        ])
        
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    times = []
    stds = []
    for i in range(3):
        # Define dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=i
        )
        
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        times.append(np.mean(res))
        stds.append(np.std(res))
            
        print('Timing:' + str(np.mean(res)) +'+-' + str(np.std(res)))
    
    fig =plt.figure()
    plt.errorbar(np.arange(3), times, yerr=stds)
    plt.show