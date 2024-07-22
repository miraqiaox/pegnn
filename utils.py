import warnings
import torch_scatter
from torch import Tensor
from typing import Tuple, Optional
from .functions.mst import mst as mst
from torch import nn
import torch
from scipy.sparse.csgraph import minimum_spanning_tree

def matrixnorm(adj):
    
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    B, N, _ = adj.size()
    adj = adj.clone()
    idx = torch.arange(N, dtype=torch.long, device=adj.device)
    adj[:, idx, idx] = 1
    deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
    adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
    return adj.squeeze(0)

def adj_to_mst(adj):

       
    adj = minimum_spanning_tree(adj)
    adj = torch.FloatTensor(adj.toarray())
    return adj

    

if __name__ == '__main__':
    pass
