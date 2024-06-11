import torch
import yaml
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
from timm.scheduler.cosine_lr import CosineLRScheduler
import argparse

from dataset.dataset_for_graph_imputed import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_huber_loss
from model.graphsamformer import GraphFormer, SAM
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--nth', default=0)
args = parser.parse_args()
def main(nth_test="all"):
    with open("./config/config_prediction.yaml", 'r') as f:
        config = yaml.safe_load(f)

    base_dir = "./log_prediction/graphformer_w_imputation/"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    check_dir(base_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, 'train_prediction_{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
    print(config)
    logging.info(config)

    if nth_test == 'all':
        index="median"
    else:
        index = str(nth_test)

    train_dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='train', index=nth_test)
    chla_scaler = train_dataset.chla_scaler
    train_dloader = DataLoader(train_dataset, config['batch_size'], shuffle=False, prefetch_factor=2, num_workers=2)
    test_dloader = DataLoader(PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='test', index=nth_test), config['batch_size'], prefetch_factor=2, num_workers=2, shuffle=False)
    adj = np.load("/home/mafzhang/data/PRE/8d/adj.npy")
    node_type = np.load("/home/mafzhang/data/PRE/8d/node_type.npy")
    adj = torch.from_numpy(adj).float().to(device)
    node_type = torch.from_numpy(node_type).long().to(device)

    mean = torch.from_numpy(train_dataset.mean).to(device)
    std = torch.from_numpy(train_dataset.std).to(device)

    model = GraphFormer(config, mean, std)
    model = model.to(device)

    chla_mae_list, chla_mse_list = [], []
    predictions = []
    trues = []
    with torch.no_grad():
        for _, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
            datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)
            prediction = model(datas, adj, node_type)
            labels = model.normalize(labels)
            mask = label_masks.cpu()
            label = labels.cpu()

            chla_mae = (torch.abs(prediction.cpu() - label) * mask).sum([1, 2, 3]) / (mask.sum([1, 2, 3]) + 1e-5)
            chla_mse = (((prediction.cpu() - label) * mask) ** 2).sum([1, 2, 3]) / (mask.sum([1, 2, 3]) + 1e-5)

            chla_mae_list.append(chla_mae)
            chla_mse_list.append(chla_mse)
            predictions.append(prediction.cpu())
            trues.append(label)

    chla_mae = torch.cat(chla_mae_list, 0)
    chla_mse = torch.cat(chla_mse_list, 0)
    chla_mae = chla_mae[chla_mae != 0].mean()
    chla_mse = chla_mse[chla_mse != 0].mean()
    predictions = torch.cat(predictions, 0)
    trues = torch.cat(trues, 0)

    log_buffer = "For dataset {}, test mae - {:.4f}, ".format(index, chla_mae)
    log_buffer += "test mse - {:.4f}".format(chla_mse)

    print(log_buffer)
    logging.info(log_buffer)
    np.save(base_dir + "prediction_{}_train.npy".format(str(index)), predictions)
    np.save(base_dir + "true_train.npy", trues)

if __name__=="__main__":
    main(args.nth)
