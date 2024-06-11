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
from utils import check_dir, masked_mae, masked_mse, seed_everything
from model.graphdiffusion import IAP_base

with open("./config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./log/graph_diffusion_v2/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
seed_everything(1234)

model = torch.load(base_dir + 'best.pt')
np.save(base_dir+"learnable_position_embedding.npy", model.diffusion_model.learnable_position_embedding.cpu().detach().numpy())
