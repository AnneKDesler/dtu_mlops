"""
LFW dataloading
"""
import argparse
import time
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.transform = transform
        self.path_to_folder = path_to_folder
        self.len = 0
        self._cum_ixs, self.labels = self._collect_file_ixs(path_to_folder)

    def _collect_file_ixs(self, path_to_folder):
        labels = os.listdir(path_to_folder)
        cum_ixs = [0]
        acc = 0
        for label in labels:
            n_files = len(os.listdir(os.path.join(path_to_folder, label)))
            acc += n_files
            cum_ixs.append(acc)
            
        return cum_ixs, labels
        
    def _find_file(self, index: int) -> int:
        assert index >= 0
        prev = -1
        for i, ix in enumerate(self._cum_ixs):
            if ix > index:
                file_no = index-prev+1  # File number in folder
                folder = self.labels[i-1]
                return os.path.join(
                    self.path_to_folder, 
                    folder, 
                    folder+"_{:04d}.jpg".format(file_no)
                )
            prev = ix
        raise ValueError("Index not found in dataset")

    def __len__(self):
        return self._cum_ixs[-1]
    
    def __getitem__(self, index: int) -> torch.Tensor:
        file = self._find_file(index)
        with Image.open(file) as img:
            return self.transform(img), file.split('\\')[-2]
        

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-path_to_folder', 
        default='..\\..\\lfw', 
        type=str
    )
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        batch = next(iter(dataloader))
        imgs = [batch[0][i,:,:,:] for i in range(args.batch_size)]
        labels = [batch[1][i] for i in range(args.batch_size)]
        plt.rcParams["savefig.bbox"] = 'tight'
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[0, i].set_title(labels[i], size=5)
        plt.show()
        
    if args.get_timing or True:
        # lets do some repetitions
        res = [ ]
        for _ in tqdm(range(5)):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')
