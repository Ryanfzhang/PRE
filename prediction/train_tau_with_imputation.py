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
from model.tau import TAU
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--nth', default=0)
args = parser.parse_args()
def main(nth_test="all"):
    with open("./config/config_prediction.yaml", 'r') as f:
        config = yaml.safe_load(f)

    base_dir = "./log_prediction/tau_w_imputation/"
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
    train_dloader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    test_dloader = DataLoader(PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='test', index=nth_test), 1, shuffle=False)

    # model = TAU(config['in_len'], config['out_len'], 1, 1, H=config['height'], W=config['width'])
    model = torch.load(base_dir + "best_tau_with_imputation_{}.pt".format(nth_test))
    model = model.to(device)

    train_process = tqdm(range(config['epochs']))
    optimizer = torch.optim.Adam(model.parameters(), config['lr'], weight_decay=config['wd'])
    optimizer_scheduler = CosineLRScheduler(optimizer, config['epochs'], lr_min=1e-6, warmup_lr_init=1e-5, t_in_epochs=True, k_decay=1.0)

    best_mae_chla = 100

    # for epoch in train_process:
    #     data_time_m = AverageMeter()
    #     losses_m = AverageMeter()
    #     model.train()
    #     optimizer_scheduler.step(epoch)
    #     end = time.time()
    #     for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
    #         datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.to(device), data_ob_masks.to(
    #             device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)
    #
    #         prediction = model(datas)
    #         loss = masked_mse(prediction, labels, label_masks)
    #         losses_m.update(loss.item(), datas.size(0))
    #         data_time_m.update(time.time() - end)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1)
    #         optimizer.step()
    #         torch.cuda.synchronize()
    #
    #     log_buffer = "train prediction loss : {:.4f}".format(losses_m.avg)
    #     log_buffer += "| time : {:.4f}".format(data_time_m.avg)
    #     train_process.write(log_buffer)
    #     losses_m.reset()
    #     data_time_m.reset()
    #     if epoch % config['test_freq'] == 0 and epoch != 0:
    chla_mae_list, chla_mse_list = [], []
    predictions = []
    trues = []
    with torch.no_grad():
        for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
            datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)
            prediction = model(datas)
            mask = label_masks.cpu()
            label = labels.cpu()

            chla_mae = (torch.abs(prediction.cpu() - label) * mask).sum([1, 2, 3, 4]) / (mask.sum([1, 2, 3, 4]) + 1e-5)
            chla_mse = (((prediction.cpu() - label) * mask) ** 2).sum([1, 2, 3, 4]) / (mask.sum([1, 2, 3, 4]) + 1e-5)

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
    if chla_mae < best_mae_chla:
        torch.save(model, base_dir + "best_tau_with_imputation_{}.pt".format(nth_test))
        np.save(base_dir + "prediction_{}.npy".format(str(index)), predictions)
        np.save(base_dir + "true.npy", trues)
        best_mae_chla = chla_mae

if __name__=="__main__":
    for i in range(20):
        main(i)
