
from sklearn.metrics.pairwise import cosine_similarity
from utils import MinimumSpanningTree, adj_to_mst
from model import FD_GCN
from data import TempDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
import random
import shutil
import yaml
import torch
import time
import os



def Train():

    validloader = DataLoader(validset, batch_size=batch_size)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    ckptpath = savepath + '/model'
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    Loss = torch.nn.L1Loss()
    scheduler = CosineAnnealingLR(
        optimizer=optimizer, T_max=epochs, eta_min=1e-9)

    mae = 999

    for epoch in range(epochs):
        TimeStart = time.time()
        totalloss = 0

        for i, (feats, y) in enumerate(trainloader):

            model.train()

            pred = model(feats.to(device), adj.to(device))
            loss = Loss(pred.squeeze(-1), y.to(device))
            optimizer.zero_grad()

            loss.backward()

            totalloss += loss.item()
            optimizer.step()
  

        scheduler.step()

        totalloss = totalloss/len(trainloader)


        if (epoch+1) % interval == 0:
            validloss = 0
            with torch.no_grad():
                model.eval()
                for _, (x, label) in enumerate(validloader):

                    pred = model(x.to(device), adj.to(device))
                    loss = Loss(pred.squeeze(-1), label.to(device))
 
                    validloss += loss.item()

            validloss = validloss / len(validloader)
            if validloss < mae:
                if os.path.getsize(ckptpath):
                    shutil.rmtree(ckptpath)
                    os.makedirs(ckptpath)
                torch.save(model, ckptpath +
                           '/model{:4d}_{:f}.pth'.format(epoch+1, validloss*15))

                mae = validloss
                best_epoch = epoch
            TimeEnd = time.time()

            print('epoch:{:d}|train_loss:{:4f}|valid_loss:{:4f}|epoch_time:{:4f}s'.format(
                epoch + 1, totalloss*15, validloss*15, TimeEnd-TimeStart))
        else:
            TimeEnd = time.time()
            print('epoch:{:d}|train_loss:{:4f}||epoch_time:{:4f}s'.format(
                epoch + 1, totalloss*15, TimeEnd-TimeStart))

            # adj = adj+adj.T+torch.eyes(adj.shape)
            # adj[adj>0]=1
    return best_epoch, mae, totalloss

if __name__ == '__main__':


    cfg = 'config/temp.yaml'
    with open(cfg, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    # config['valindex'] = ValIndex
    batch_size = config['batch_size']
    epochs = config['epoch']
    # epochs = 10
    device = config['device']
    learning_rate = config['learning_rate']
    layers = config['layer']
    hidden = config['hidden']
    interval = config['interval']
    savepath = config['save_path'] + '/pegnn/rotate/cons_5000/5/' + \
        datetime.now().strftime("%Y%m%d%H%M%S") + '_gelu'
    os.makedirs(savepath)
    with open(savepath + '/config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    obs_axis = np.array([[80, 25], [50, 10], [80, 80], [20, 80], [21, 15], [21, 35], [30, 52], [
                        50, 80], [81, 55], [50, 30], [25, 25], [60, 25], [20, 63], [75, 65], [60, 50], [50, 63]])



    trainset = TempDataset('./comp_fusion.pickle', obs_axis=obs_axis, index=[
        i for i in range(5000)], mean=300, std=15)
    validset = TempDataset('./comp_fusion.pickle', obs_axis=obs_axis, index=[
        i for i in range(1000)], mean=300, std=15)
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

    model = FD_GCN(2, len(obs_axis), 16, 16, 10000, 1, 'gelu').to(device)


 
    print('start training')
    best_epoch, best_loss, train_loss = Train()

