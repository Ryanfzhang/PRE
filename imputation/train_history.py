import argparse
import torch
import datetime
import json
import yaml
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
from timm.scheduler.cosine_lr import CosineLRScheduler

from dataset.dataset import PREDataset, PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_cor
from model.diffusion import IAP_base

with open("./config_mae.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./runs_8d/his/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(base_dir, '{}.log'.format(timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)


train_dataset = PRE8dDataset(data_root='./data/PRE/8d', in_len=config['in_len'], 
                             out_len=config['out_len'], missing_ratio=config["missing_ratio"],
                         mode='train')
chla_scaler = train_dataset.chla_scaler
test_dloader = DataLoader(PRE8dDataset(data_root='./data/PRE/8d', in_len=config['in_len'], 
                        out_len=config['out_len'], missing_ratio=config["missing_ratio"],
                         mode='test'), 1, shuffle=False)

chla_mae_list, chla_mse_list= [], []
for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
    datas , data_ob_masks, data_gt_masks, labels, label_masks = datas.to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)

    mean_datas = train_dataset.imputation[648-config['in_len']-config['out_len']+test_step:648-config['out_len']+test_step]
    mean_datas = torch.from_numpy(mean_datas).to(device)
    imputed_data = torch.where(data_gt_masks.bool(), datas, mean_datas)
    imputed_data = imputed_data.cpu()

    mask = (data_ob_masks - data_gt_masks).cpu()
    chla_mae= masked_mae(imputed_data[:,:,0], datas[:,:,0].cpu(), mask[:,:,0])
    chla_mse= masked_mse(imputed_data[:,:,0], datas[:,:,0].cpu(), mask[:,:,0])
    chla_mae_list.append(chla_mae)
    chla_mse_list.append(chla_mse)

chla_mae = torch.stack(chla_mae_list, 0)
chla_mse = torch.stack(chla_mse_list, 0)

log_buffer = "test mae: chla-{:.4f}, ".format(chla_mae[chla_mae!=0].mean())
log_buffer += "test mse: chla-{:.4f}".format(chla_mse[chla_mse!=0].mean())
print(log_buffer)
logging.info(log_buffer)
