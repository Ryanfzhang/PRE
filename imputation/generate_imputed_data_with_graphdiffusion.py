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
import numpy as np

from dataset.dataset_for_graph import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse
from model.graphdiffusion_v3 import IAP_base
from utils import seed_everything

with open("./config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

seed_everything(1234)

base_dir = "./log/graph_diffusion_v3/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)

model = torch.load(base_dir+'best_0.1.pt')
model = model.to(device)
print(model)
logging.info(model)
datapath = "/home/mafzhang/data/PRE/8d/missing_0.1_in_46_out_46_1.pk"
if os.path.isfile(datapath) is False:
    print("file does not exist")
    exit()
with open(datapath,'rb') as f:
    datas, data_ob_masks, data_gt_masks, labels, label_ob_masks = pickle.load(
                    f
                )

is_sea = np.load("/home/mafzhang/data/PRE/8d/is_sea.npy").astype(bool)
adj = np.load("/home/mafzhang/data/PRE/8d/adj.npy")
node_type = np.load("/home/mafzhang/data/PRE/8d/node_type.npy")
adj = torch.from_numpy(adj).float().to(device)
node_type = torch.from_numpy(node_type).long().to(device)


bs = 24
step = datas.shape[0]//bs + 1
num_samples = 10

imputed_datas=[]
for i in tqdm(range(step)):
    data = datas[bs*i:min(bs*i+bs, datas.shape[0])]
    data_mask = data_ob_masks[bs*i:min(bs*i+bs, datas.shape[0])]


    data_graph = torch.from_numpy(data[:,:,:,is_sea]).float().to(device)
    data_mask_graph = torch.from_numpy(data_mask[:,:,:,is_sea]).to(device)

    imputed_data = model.impute(data_graph, data_mask_graph, adj, node_type, 10)
    data_mask_graph = data_mask_graph.unsqueeze(1).expand_as(imputed_data)
    data_graph = data_graph.unsqueeze(1).expand_as(imputed_data)
    imputed_data = data_mask_graph.cpu()*data_graph.cpu() + (1-data_mask_graph.cpu())*imputed_data
    imputed_datas.append(imputed_data)

imputed_datas_graph = torch.cat(imputed_datas,dim=0)
imputed_datas = torch.zeros(imputed_datas_graph.shape[0],10,46,1,60,96)
imputed_datas[:,:,:,:,is_sea]=imputed_datas_graph
new_data_path="/home/mafzhang/data/PRE/8d/missing_0.1_in_46_out_46_1_imputed_graph.pk"
with open(new_data_path, 'wb') as f:
    pickle.dump([imputed_datas.numpy(), data_ob_masks,data_gt_masks,labels,label_ob_masks], f)
