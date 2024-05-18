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

from dataset.dataset import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse
from model.diffusion import IAP_base

with open("./config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./log/diffusion_v2/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
print(config)


train_dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='train')

chla_scaler = train_dataset.chla_scaler

model = torch.load(base_dir+'best.pt')
model = model.to(device)
index = [i*12 for i in range(52)]
datas, data_ob_masks, data_gt_masks, _, _ = train_dataset.__getitem__(index)
datas, data_ob_masks = torch.from_numpy(datas).to(device), torch.from_numpy(data_ob_masks).to(device)
imputed_data = model.impute(datas, data_ob_masks, config['num_samples'])
imputed_data = imputed_data.median(dim=1).values

np.save("./imputation_test/datas.npy", datas.cpu().numpy())
np.save("./imputation_test/imputed_x_diffusion.npy", imputed_data.numpy())
np.save("./imputation_test/data_ob_masks.npy", data_ob_masks.cpu().numpy())
np.save("./imputation_test/data_gt_masks.npy", data_gt_masks)

