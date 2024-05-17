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

from dataset.dataset import PREDataset,PRE8dDataset
from utils import check_dir, masked_mae, masked_mse
from model.diffusion import IAP_base

with open("./config.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./runs_8d/xdiffusion/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)


train_dataset = PRE8dDataset(data_root='./data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='train')
chla_scaler = train_dataset.chla_scaler

model = torch.load(base_dir+'best.pt')
model = model.to(device)

index = [i*12 for i in range(16)]
datas, data_ob_masks, data_gt_masks, _, _ = train_dataset.__getitem__(index)
datas, data_ob_masks = torch.from_numpy(datas).to(device), torch.from_numpy(data_ob_masks).to(device)
imputed_data = model.impute(datas, data_ob_masks, config['num_samples'])
imputed_data = imputed_data.median(dim=1).values
imputed_data = torch.where(data_ob_masks.bool().cpu(), datas.cpu(), imputed_data)

np.save("./imputed_x.npy", imputed_data.numpy())
np.save("./data_ob_masks.npy", data_ob_masks.cpu().numpy())
np.save("./data_gt_masks.npy", data_gt_masks)

