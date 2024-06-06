import yaml
import pickle
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch
from einops import rearrange

from dataset.dataset import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse
from model.diffusion import IAP_base
from utils import seed_everything

with open("./config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

seed_everything(1234)

base_dir = "./log/grin/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
print(config)

model = torch.load(base_dir+'best.pt')
model = model.to(device)
print(model)
datapath = "/home/mafzhang/data/PRE/8d/missing_0.1_in_12_out_1_1.pk"
if os.path.isfile(datapath) is False:
    print("file does not exist")
    exit()
with open(datapath,'rb') as f:
    datas, data_ob_masks, data_gt_masks, labels, label_ob_masks = pickle.load(
        f
    )

bs = 8
step = datas.shape[0]//bs + 1

imputed_datas=[]
with torch.no_grad():
    for i in tqdm(range(step)):
        data = datas[bs*i:min(bs*i+bs, datas.shape[0])]
        data_mask = data_gt_masks[bs*i:min(bs*i+bs, datas.shape[0])]
        data = torch.from_numpy(data).float().to(device)
        data_mask = torch.from_numpy(data_mask).float().to(device)
        data = rearrange(data, "b t c h w -> b t (h w) c")
        data_mask = rearrange(data_mask, "b t c h w -> b t (h w) c")

        imputed_data = model.impute(data, data_mask.bool())
        imputed_data = data_mask.cpu()*data.cpu() + (1-data_mask.cpu())*imputed_data.cpu()
        imputed_data = rearrange(imputed_data, "b t (h w) c-> b t c h w", h=config['height'], w=config['width'], c=1)
        imputed_datas.append(imputed_data)

imputed_datas = torch.cat(imputed_datas,dim=0)
new_data_path="/home/mafzhang/data/PRE/8d/missing_0.1_in_12_out_12_1_imputed_grin_gt.pk"
with open(new_data_path, 'wb') as f:
    pickle.dump([imputed_datas.numpy(), data_ob_masks,data_gt_masks,labels,label_ob_masks], f)
