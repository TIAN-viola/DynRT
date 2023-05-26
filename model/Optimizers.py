import torch

def build_Adam(params,lr,weight_decay):
    return torch.optim.Adam(params=params,lr=lr,weight_decay=weight_decay)