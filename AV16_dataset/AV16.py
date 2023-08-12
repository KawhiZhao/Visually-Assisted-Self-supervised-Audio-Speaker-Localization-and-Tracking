
import torch

from torch.utils.data import Dataset

import numpy as np
import torchvision.transforms as transforms
import pdb
from tqdm import tqdm

data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class AV16_dataset_gccphat_seg(Dataset):
    def __init__(self, data_split) -> None:
        super().__init__()
        self.width = 360
        self.height = 288
        self.data_split = data_split
        self.data_path = ''
        if self.data_split == 'train':
            self.data_path = '/mnt/fast/nobackup/scratch4weeks/jz01019/Teacher-Student/AV16_dataset/new_train.txt'
        elif self.data_split == 'eval':
            self.data_path = '/mnt/fast/nobackup/scratch4weeks/jz01019/Teacher-Student/AV16_dataset/new_eval.txt'
        elif self.data_split == 'test':
            self.data_path = '/mnt/fast/nobackup/scratch4weeks/jz01019/Teacher-Student/AV16_dataset/new_test.txt'
        
        with open(self.data_path, 'r') as f:
            self.data = f.readlines()
        
    
        
                    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path = self.data[index].strip()
        saved_data = np.load(data_path, allow_pickle=True)
        
        gcc, img, seg, gt, seq_name = tuple(saved_data)

        gt = np.array(gt)
        gt[0] = gt[0] / self.width
        gt[1] = gt[1] / self.height
        img = data_transform(img)

        return torch.tensor(gcc, dtype=torch.float32), img, torch.tensor(seg, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32), seq_name
        # convert to tensor
