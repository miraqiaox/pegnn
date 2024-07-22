import os

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import TempDataset
from model import FD_GCN
from utils import MinimumSpanningTree, adj_to_mst
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def test(net):
    testloader = DataLoader(testset, batch_size=1)
    validloss = 0
    Loss = torch.nn.L1Loss()
    with torch.no_grad():
        net.eval()
        for _, (x, label) in enumerate(testloader):

            y_hat = net(x.to(device), adj.to(device))*15 + 300
            label = label*15 + 300
            
            loss = Loss(y_hat.squeeze(), label.to(device).squeeze())

            validloss += loss.item()

    validloss = validloss / len(testloader)

    print('test_loss:{:4f}'.format(validloss))




if __name__ == '__main__':

        
        model_path = './model/' + os.listdir('./model')[0]
        cfg = './config.yaml'
        with open(cfg, encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        device = config['device']
        model = torch.load(model_path).to(device)

        obs_axis = np.array([[80, 25], [50, 10], [80, 80], [20, 80], [21, 15], [21, 35], [30, 52], [
                            50, 80], [81, 55], [50, 30], [25, 25], [60, 25], [20, 63], [75, 65], [60, 50], [50, 63]])
        t_r = 48
        # 数据集定义
        mst1 = MinimumSpanningTree()

        trainset = TempDataset('./comp_fusion.pickle', obs_axis=obs_axis, index=[
            i for i in range(5000)], mean=300, std=15)

        testset = TempDataset('./comp_test.pickle', obs_axis=obs_axis, index=[
            i for i in range(500)], mean=300, std=15)
        axis = trainset.get_axis()
        data = trainset.get_data()

        adj_d1 = torch.cdist(axis, axis)
        adj_d1 = adj_to_mst(adj_d1).to(device)

        adj_d2 = torch.cdist(axis, axis, p=1)
        adj_d2 = adj_to_mst(adj_d2).to(device)

        adj_c = torch.FloatTensor(cosine_similarity(data.T, data.T))
        adj_c = 1/torch.abs(adj_c)
        adj_c = adj_to_mst(adj_c).to(device)

        adj_r = torch.FloatTensor(np.corrcoef(data.T))
        adj_r = 1/torch.abs(adj_r)
        adj_r = adj_to_mst(adj_r).to(device)

        adj = adj_d1+adj_d2+adj_c+adj_r

        print('start testing')
        test(model)

        print('finish testing')
