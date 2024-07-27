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
print(chla.shape)
is_sea = np.sum(~np.isnan(chla), 0) >0
n_nodes = np.sum(is_sea)

chla_mask = ~np.isnan(chla)
chla_scaler = LogScaler()
chla = chla_scaler.transform(chla)
mean = np.nanmean(chla[:648], axis=0)[np.newaxis,:,:]
chla = np.nan_to_num(chla, nan=0.)
chla = chla_mask.astype(float)*chla + (1-chla_mask.astype(float))*mean
chla = np.nan_to_num(chla, nan=0.)
chla = chla[:648, is_sea.astype(bool)]

node_type = np.ones(n_nodes)
node_type[chla.min(axis=0)>-0.5]=0
node_type[chla.max(axis=0)<0.5]=2
np.set_printoptions(threshold=np.inf)


np.save("/home/mafzhang/data/PRE/8d/node_type.npy", node_type)
