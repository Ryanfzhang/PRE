import warnings
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
import h5py
import os
import pickle
from utils import StandardScaler, LogScaler
import copy
warnings.filterwarnings("ignore")

data = h5py.File(osp.join("/home/mafzhang/data/PRE/8d/modis_chla_8d_4km_pre.mat"))
chla = np.array(data["CHLA_pre"])
is_sea = np.sum(~np.isnan(chla), 0) >0
n_nodes = np.sum(is_sea)

index = np.zeros((60,96))
index[is_sea] = np.arange(1, np.sum(is_sea)+1)

source = []
target = []

for i in range(60):
    for j in range(96):
        if is_sea[i, j]:
            source.append(index[i,j]-1)
            target.append(index[i,j]-1)
            if i>0 and is_sea[i-1, j]:
                source.append(index[i, j] - 1)
                target.append(index[i-1, j] - 1)
            if i<59 and is_sea[i+1, j]:
                source.append(index[i, j] - 1)
                target.append(index[i+1, j] - 1)
            if j>0 and is_sea[i, j-1]:
                source.append(index[i, j] - 1)
                target.append(index[i, j-1] - 1)
            if j<95 and is_sea[i, j+1]:
                source.append(index[i, j] - 1)
                target.append(index[i, j+1] - 1)


adj = np.zeros((n_nodes, n_nodes))

for i in range(len(source)):
    s = int(source[i])
    t = int(target[i])
    adj[s,t] = 1

np.save("/home/mafzhang/data/PRE/8d/adj.npy", adj)