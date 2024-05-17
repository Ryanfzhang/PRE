import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
from timm.scheduler.cosine_lr import CosineLRScheduler
from dataset.dataset import PREDataset,PRE8dDataset
from xgboost import XGBRegressor

from utils import check_dir, masked_mae, masked_mse, masked_cor
from model.diffusion import IAP_base
from model.mae import MaskedAutoEncoder

with open("./config_mae.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./runs_8d/xgboost/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(base_dir, '{}.log'.format(timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)
base_dir = "./runs_8d/unified_diffusion_v2/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)
train_dataset = PRE8dDataset(data_root='./data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='train')
chla_scaler = train_dataset.chla_scaler
train_dloader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
test_dloader = DataLoader(PRE8dDataset(data_root='./data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='test'), 1, shuffle=False)

model = XGBRegressor()

train_datas = []
train_labels = []
for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
    datas = torch.permute(datas, (0, 3, 4, 1, 2))
    labels = torch.permute(labels, (0, 3, 4, 1, 2))
    labels = labels[:,:,:,:,0]
    masks = torch.permute(data_ob_masks, (0, 3, 4, 1, 2))
    datas = datas.reshape(-1, config['in_len']*1)
    labels = labels.reshape(-1, config["out_len"]*1)
    train_datas.append(datas.numpy())
    train_labels.append(labels.numpy())


train_datas = np.concatenate(train_datas, axis=0)
train_labels = np.concatenate(train_labels, axis=0)
print("Train start")
model.fit(train_datas, train_labels)
print("Test start")

sst_mae_list, sst_mse_list = [], []
chla_mae_list, chla_mse_list = [], []
with torch.no_grad():
    for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):

        b, t, c, h, w = datas.shape
        datas = torch.permute(datas, (0, 3, 4, 1, 2))
        datas = datas.reshape(-1, t*c)
        prediction = model.predict(datas)
        prediction = prediction.reshape(b, h, w, config['out_len'], -1)
        prediction = np.moveaxis(prediction, [-2, -1], [1,2])
        prediction = torch.from_numpy(prediction)

        mask = (data_ob_masks - data_gt_masks).cpu()
        chla_mae= masked_mae(prediction[:,:,0], labels[:,:,0], label_masks[:,:,0])
        chla_mse= masked_mse(prediction[:,:,0], labels[:,:,0], label_masks[:,:,0])
        chla_mae_list.append(chla_mae)
        chla_mse_list.append(chla_mse)

chla_mae = torch.stack(chla_mae_list, 0)
chla_mse = torch.stack(chla_mse_list, 0)
chla_mae = chla_mae[chla_mae!=0].mean()
chla_mse = chla_mse[chla_mse!=0].mean()

log_buffer = "test mae - {:.4f}, ".format(chla_mae)
log_buffer += "test mse - {:.4f}".format(chla_mse)
print(log_buffer)
logging.info(log_buffer)
