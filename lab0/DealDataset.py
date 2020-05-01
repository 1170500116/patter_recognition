import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
import numpy as np
from  torch.autograd import Variable

import load_minist_data


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = load_minist_data.load_data(folder, data_name, label_name) # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

    def __getall__(self):
        img = []
        target = []
        for index in range(len(self)):
            img.append(self.__getitem__( index)[0])
            target.append(self.__getitem__( index)[1])

        return img, target
