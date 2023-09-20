import os

import pandas as pd
import statistics
import math

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

class SCUT_FBP5500_Dataset(Dataset):
    def __init__(self, data_dir, split_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.split_path = os.path.join(self.data_dir, split_path)

        self.rate_detail, self.std = {}, {}
        with open(os.path.join(data_dir, "Rate_details.txt"), 'r') as f:
            info = f.readlines()
        for i in info:
            i = i.split(' ')
            if i[0] in self.rate_detail:
                self.rate_detail[i[0]].append(int(i[1]))
            else:
                self.rate_detail[i[0]] = [int(i[1])]

        for k, v in self.rate_detail.items():
            self.std[k] = statistics.variance(v)

        self.x = []
        self.y = []
        with open(self.split_path, 'r') as f:
            infos = f.readlines()
        for info in infos:
            img, label = info.split(" ")
            self.x.append(img)
            self.y.append((float(label) - 1) * 20)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, 'Images', str(self.x[idx])))
        if self.transform is not None:
            img = self.transform(img)
        label = self.y[idx]
        s = self.std[self.x[idx]]
        dist = torch.distributions.Laplace(label / 20, s)
        x = torch.linspace(1, 5, 81)
        std = dist.log_prob(x).exp()
        std = F.normalize(std, p=1, dim=0)
        return img, label, std
