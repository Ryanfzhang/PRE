import argparse
import torch
import datetime
import json
import yaml
import os
from torch.utils.data import DataLoader
import logging
import time
import numpy as np
from tqdm import tqdm
from timm.utils import AverageMeter
from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange

from dataset.dataset import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_cor
from model.diffusion import IAP_base
from model.mae import MaskedAutoEncoder
from model.grin import GRINet

with open("./config/config_mae.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./log/grin/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(base_dir, '{}.log'.format(timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)


train_dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d', in_len=config['in_len'], 
                        out_len=config['out_len'],missing_ratio=config['missing_ratio'],
                         mode='train')
chla_scaler = train_dataset.chla_scaler
train_dloader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
test_dloader = DataLoader(PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d', in_len=config['in_len'], 
                        out_len=config['out_len'],missing_ratio=config['missing_ratio'],
                         mode='test'), 1, shuffle=False)

adj = torch.zeros([config['height']*config['width'], config['height']*config['width']])
for i in range(0, config['height']):
    for j in range(0, config['width']):
        adj[i*96+j,i*96+min(j+1, 95)] = 1.
        adj[i*96+j,i*96+max(j-1, 0)] = 1.
        adj[i*96+j,i*96+j] = 1.
        adj[i*96+j,(max(i-1, 0))*96+j] = 1.
        adj[i*96+j,(min(i+1,59))*96+j] = 1.

model = GRINet(config=config, adj=adj, d_in=1, d_hidden=32, d_ff=32, ff_dropout=0.)
model = model.to(device)
train_process = tqdm(range(config['epochs']))
optimizer = torch.optim.Adam(model.parameters(), config['lr'], weight_decay=config['wd'])
optimizer_scheduler = CosineLRScheduler(optimizer, config['epochs'], lr_min=1e-6, warmup_lr_init=1e-5, t_in_epochs=True, k_decay=1.0)

best_mae_sst = 100
best_mae_chla = 100
for epoch in train_process:
    data_time_m = AverageMeter() 
    losses_m = AverageMeter()
    model.train()
    optimizer_scheduler.step(epoch)
    end = time.time()
    for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
        datas , data_ob_masks, data_gt_masks, labels, label_masks = datas.to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)
        datas = rearrange(datas, "b t c h w -> b t (h w) c")
        ob_masks = rearrange(data_ob_masks, "b t c h w -> b t (h w) c")
        loss= model.trainstep(datas, ob_masks, labels, label_masks, is_train=1)
        losses_m.update(loss.item(), datas.size(0))
        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        # torch.cuda.synchronize()

        log_buffer = "train loss : {:.4f}".format(loss.item())
        log_buffer +=  "| time : {:.4f}".format(data_time_m.avg)
        end = time.time()
        train_process.set_description(log_buffer)

    if epoch % config['test_freq'] == 0:
        chla_mae_list, chla_mse_list= [], []
        imputed_data_list = []
        with torch.no_grad():
            for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
                datas , data_ob_masks, data_gt_masks, labels, label_masks = datas.to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)

                ob_datas = rearrange(datas, "b t c h w -> b t (h w) c")
                gt_masks = rearrange(data_gt_masks, 'b t c h w -> b t (h w) c')
                imputed_data = model.impute(ob_datas, gt_masks.bool())
                imputed_data = imputed_data.detach().cpu()
                imputed_data = rearrange(imputed_data, "b t (h w) c-> b t c h w", h=config['height'], w=config['width'], c=1)

                mask = (data_ob_masks - data_gt_masks).cpu()
                chla_mae= masked_mae(imputed_data[:,:,0], datas[:,:,0].cpu(), mask[:,:,0])
                chla_mse= masked_mse(imputed_data[:,:,0], datas[:,:,0].cpu(), mask[:,:,0])
                chla_mae_list.append(chla_mae)
                chla_mse_list.append(chla_mse)
                imputed_data_list.append(imputed_data)

        chla_mae = torch.stack(chla_mae_list, 0)
        chla_mse = torch.stack(chla_mse_list, 0)
        chla_mae = chla_mae[chla_mae!=0].mean()
        chla_mse = chla_mse[chla_mse!=0].mean()
        imputed_datas = torch.cat(imputed_data_list,dim=0)

        log_buffer = "test mae: chla-{:.4f}, ".format(chla_mae)
        log_buffer += "test mse: chla-{:.4f}".format(chla_mse)
        print(log_buffer)
        logging.info(log_buffer)
        if chla_mae<best_mae_chla:
            torch.save(model, base_dir+'best.pt')
            np.save(base_dir+'imputed_x.pt', imputed_datas.numpy())
