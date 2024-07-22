import torch
import pickle
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.distributions import Normal

class TempDataset(Dataset):
    def __init__(self, root='./comp_fusion.pickle', obs_axis=[1, 2], index=[i for i in range(100)],  mean=0., std=1.):
        with open(root, 'rb') as df:
            temp = torch.FloatTensor(pickle.load(df)).T.reshape(-1, 100, 100)
            # temp = torch.FloatTensor(pickle.load(df))
        temp = (temp - mean) / std
        self.data = temp[index]
        sensor = np.array(obs_axis)
        sparse_data = []
        for i in range(sensor.shape[0]):
            sparse_data.append(self.data[:, sensor[i, 0], sensor[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        h = 100
        x = torch.linspace(1, h, h).unsqueeze(0).repeat(h, 1).unsqueeze(-1)
        y = torch.linspace(1, h, h).unsqueeze(1).repeat(1, h).unsqueeze(-1)
        self.axis = torch.cat((x, y), dim=-1).transpose(1, 0).reshape(-1, 2)
        # self.data = torch.FloatTensor(data[:, :, 3:]) 

    def __getitem__(self, index):
        return self.observe[index], self.data[index].reshape(-1)

    def __len__(self):
        return self.data.shape[0]

    def get_axis(self):
        return self.axis

    def get_data(self):
        return self.data.reshape(-1, 10000)
    



if __name__ == '__main__':
    pass

