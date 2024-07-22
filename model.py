import math
import torch
import copy
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Any, Optional
from torch.nn.parameter import Parameter
from utils import matrixnorm
from torch.utils.checkpoint import checkpoint
from torch.nn import Module, ModuleList, LeakyReLU, InstanceNorm1d, Conv1d, BatchNorm1d






def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == "gelu":
        layer = nn.GELU()
    elif act == "sigmoid":
        layer = nn.Sigmoid()
    elif act == "tanh":
        layer = nn.Tanh()
    else:
        raise NotImplementedError("activation layer [%s] is not found" % act)
    return layer

    

class GradientOperator(nn.Module):
    def __init__(
        self, node_num=4, node_feat=4, grad_feats=4
    ):
        super(GradientOperator, self).__init__()

        self.w1 = nn.parameter.Parameter(torch.rand(node_num, node_feat, grad_feats))
        torch.nn.init.kaiming_normal_(self.w1.data, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, adj):
        out = torch.einsum('bnf, nfg, nn->bng', x, self.w1, adj)
        return out
    

class GraphConvLayer(nn.Module):
    def __init__(
        self, input_dim=4, output_dim=4, node_num=4, h=1, act='tanh'
    ):
        super(GraphConvLayer, self).__init__()



        self.h = h
        self.grad = GradientOperator(node_num, input_dim, output_dim)
        self.K = nn.Linear(output_dim, output_dim)
        self.act = act_layer(act)
        self.K_t = nn.Linear(output_dim, output_dim)
        self.grad_t = GradientOperator(node_num, output_dim, input_dim)
        self.norm = nn.BatchNorm1d(node_num)
    def forward(self, x, adj):

        out = self.grad(x, adj)


        out = self.act(self.K(out))

        out = self.K_t(out)

        out = self.act(self.grad_t(out, adj))
        out = x - self.h*out

        return out
    

class FD_GCN(nn.Module):
    def __init__(
        self, nlayers=4, sensor_num=4, hidden_dim=4, conv_dim=4, node_num=4, h=1, act='tanh'
    ):
        super(FD_GCN, self).__init__()
        

        self.lin = nn.Linear(sensor_num, node_num)
        self.input = nn.Linear(1, hidden_dim)
        self.layerlist = ModuleList()

        for i in range(nlayers):
            self.layerlist.append(
                GraphConvLayer(hidden_dim, conv_dim, node_num, h, act)
            )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj):
        
        adj = matrixnorm(adj)
        x = self.lin(x).unsqueeze(-1)
        x = F.leaky_relu(self.input(x))
        for _, layer in enumerate(self.layerlist):
            x = layer(x, adj)
        pred = self.out(x)
        return pred.squeeze()

if __name__ == '__main__':
    pass
