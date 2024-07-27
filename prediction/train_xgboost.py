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

from dataset_pre.dataset_for_graph_imputed import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_huber_loss
from xgboost import XGBRegressor
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--nth', default=0)
args = parser.parse_args()
def main(nth_test="all"):
    with open("./config/config_prediction.yaml", 'r') as f:
        config = yaml.safe_load(f)

    base_dir = "./log_prediction/xgboost_w_imputation/input_{}_output_{}".format(str(config['in_len']), str(config['out_len']))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    check_dir(base_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, 'train_prediction_{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
    print(config)
    logging.info(config)

    train_dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='train', index=nth_test)
    train_dloader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    test_dloader = DataLoader(PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='test', index=nth_test), 1, shuffle=False)
    mean = train_dataset.mean[np.newaxis, np.newaxis, np.newaxis,:]
    std = train_dataset.std[np.newaxis, np.newaxis, np.newaxis,:]
    index = str(nth_test)

    model = XGBRegressor()

    best_mae_chla = 100
    train_datas = []
    train_labels = []
    for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
        datas = datas- mean
        labels = labels - mean
        datas = torch.permute(datas, (0, 3, 1, 2)).numpy()
        labels = torch.permute(labels, (0, 3, 1, 2)).numpy()
        labels = labels[:, :, :, 0]
        masks = torch.permute(data_ob_masks, (0, 3, 1, 2))
        datas = datas.reshape(-1, config['in_len'] * 1)
        labels = labels.reshape(-1, config["out_len"] * 1)
        train_datas.append(datas)
        train_labels.append(labels)

    train_datas = np.concatenate(train_datas, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    print("For dataset {}, Train start".format(str(index)))
    model.fit(train_datas, train_labels)
    print("For dataset {}, Test start".format(str(index)))

    predictions = []
    labels_list = []
    label_masks_list = []
    with torch.no_grad():
        for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
            b, t, c, n = datas.shape
            datas = datas - mean
            datas = torch.permute(datas, (0, 3, 1, 2))
            datas = datas.reshape(-1, t * c)
            prediction = model.predict(datas)
            prediction = prediction.reshape(b, n, 1, -1)
            prediction = np.moveaxis(prediction, [1, -1], [-1, 1])  # b t c n
            prediction = prediction + mean
            prediction = torch.from_numpy(prediction)
            mask = label_masks.cpu()
            label = labels.cpu()

            predictions.append(prediction.cpu())
            labels_list.append(label)
            label_masks_list.append(mask)

        predictions = torch.cat(predictions, 0)
        labels = torch.cat(labels_list, 0)
        label_masks = torch.cat(label_masks_list, 0)
        chla_mae = (torch.abs(predictions - labels) * label_masks).sum([1, 2]) / (label_masks.sum([1, 2]) + 1e-5)
        chla_mse = (((predictions - labels) * label_masks) ** 2).sum([1, 2]) / (label_masks.sum([1, 2]) + 1e-5)
        chla_mae = chla_mae.mean(0)
        chla_mse = chla_mse.mean(0)

        chla_mae_mean = chla_mae[chla_mae!=0].mean()
        chla_mse_mean = chla_mse[chla_mse!=0].mean()

        log_buffer = "For dataset , test mae - {:.4f}, ".format(chla_mae_mean)
        log_buffer += "test mse - {:.4f}".format(chla_mse_mean)

        print(log_buffer)
        logging.info(log_buffer)
        np.save(base_dir + "prediction_{}.npy".format(index), predictions)

if __name__=="__main__":
    main(args.nth)
