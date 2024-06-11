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

from dataset.dataset_imputed import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_huber_loss
from xgboost import XGBRegressor
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--nth', default=0)
args = parser.parse_args()
def main(nth_test="all"):
    with open("./config/config_prediction.yaml", 'r') as f:
        config = yaml.safe_load(f)

    base_dir = "./log_prediction/xgboost_w_imputation/"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    check_dir(base_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, 'train_prediction_{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
    print(config)
    logging.info(config)

    train_dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='train', index=nth_test)
    chla_scaler = train_dataset.chla_scaler
    train_dloader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    test_dloader = DataLoader(PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='test', index=nth_test), 1, shuffle=False)
    if nth_test == 'all':
        index="median"
    else:
        index = str(nth_test)

    model = XGBRegressor()

    best_mae_chla = 100
    train_datas = []
    train_labels = []
    for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
        datas = torch.permute(datas, (0, 3, 4, 1, 2)).numpy()
        labels = torch.permute(labels, (0, 3, 4, 1, 2)).numpy()
        labels = labels[:, :, :, :, 0]
        masks = torch.permute(data_ob_masks, (0, 3, 4, 1, 2))
        datas = datas.reshape(-1, config['in_len'] * 1)
        labels = labels.reshape(-1, config["out_len"] * 1)
        train_datas.append(datas)
        train_labels.append(labels)

    train_datas = np.concatenate(train_datas, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    print("For dataset {}, Train start".format(str(index)))
    model.fit(train_datas, train_labels)
    print("For dataset {}, Test start".format(str(index)))

    chla_mae_list, chla_mse_list = [], []
    predictions = []
    trues = []
    with torch.no_grad():
        for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
            b, t, c, h, w = datas.shape
            datas = torch.permute(datas, (0, 3, 4, 1, 2))
            datas = datas.reshape(-1, t * c)
            prediction = model.predict(datas)
            prediction = prediction.reshape(b, h, w, config['out_len'], -1)
            prediction = np.moveaxis(prediction, [-2, -1], [1, 2])  # b t c h w
            prediction = torch.from_numpy(prediction)
            mask = label_masks.cpu()
            label = labels.cpu()

            chla_mae = (torch.abs(prediction.cpu() - label) * mask).sum([1, 2, 3, 4]) / (mask.sum([1, 2, 3, 4]) + 1e-5)
            chla_mse = (((prediction.cpu() - label) * mask) ** 2).sum([1, 2, 3, 4]) / (mask.sum([1, 2, 3, 4]) + 1e-5)
            chla_mae_list.append(chla_mae)
            chla_mse_list.append(chla_mse)
            predictions.append(prediction)
            trues.append(labels)


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
        if chla_mae < best_mae_chla:
            np.save(base_dir + "prediction_{}.npy".format(str(index)), predictions)
            np.save(base_dir + "true.npy", trues)
            best_mae_chla = chla_mae

if __name__=="__main__":
    main(args.nth)