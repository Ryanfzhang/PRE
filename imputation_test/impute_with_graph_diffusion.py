import numpy
import torch
import yaml
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np

from dataset.dataset_for_graph import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse
from model.diffusion import IAP_base

with open("./config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./log/graph_diffusion/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
print(config)


train_dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='train')

chla_scaler = train_dataset.chla_scaler
adj = np.load("/home/mafzhang/data/PRE/8d/adj.npy")
node_type = np.load("/home/mafzhang/data/PRE/8d/node_type.npy")
adj = torch.from_numpy(adj).float().to(device)
adj = adj - torch.eye(adj.shape[0]).to(device)
node_type = torch.from_numpy(node_type).long().to(device)

model = torch.load(base_dir+'best.pt')
model = model.to(device)
saves = []
for t in range(4):
    index = [i*12 for i in range(t*4,t*4+4)]
    datas, data_ob_masks, data_gt_masks, _, _ = train_dataset.__getitem__(index)
    datas, data_ob_masks = torch.from_numpy(datas).to(device), torch.from_numpy(data_ob_masks).to(device)
    imputed_data = model.impute(datas, data_ob_masks,adj, node_type, config['num_samples'])
    imputed_data = imputed_data.median(dim=1).values
    B, T, C, N = imputed_data.shape
    save = torch.zeros((B,T,C,60,96))
    save[:,:,:, ~train_dataset.is_land.astype(bool)]=imputed_data
    saves.append(save)
saves = torch.cat(saves,dim=0)

np.save("./imputation_test/imputed_x_graphdiffusion.npy", saves.numpy())
