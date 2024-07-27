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
from einops import rearrange

from dataset_pre.dataset_for_graph_imputed import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_huber_loss, huber_loss, mse_loss
from model.graphTMixer import GraphTMixer
from model.graphsamformer import GraphFormer
from torchtsmixer import TSMixer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--nth', default=0)
args = parser.parse_args()
def main(nth_test="all"):
    with open("./config/config_prediction.yaml", 'r') as f:
        config = yaml.safe_load(f)

    base_dir = "./log_prediction/tsmixer_w_imputation/input_{}_output_{}".format(str(config['in_len']), str(config['out_len']))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    check_dir(base_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, 'train_prediction_{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
    print(config)
    logging.info(config)

    index = str(nth_test)

    train_dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='train', index=nth_test)
    train_dloader = DataLoader(train_dataset, config['batch_size'], shuffle=True, prefetch_factor=2, num_workers=2)
    test_dloader = DataLoader(PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='test', index=nth_test), config['batch_size'], prefetch_factor=2, num_workers=2, shuffle=False)
    adj = train_dataset.adj
    node_type = np.load("/home/mafzhang/data/PRE/8d/node_type.npy")
    adj = torch.from_numpy(adj).float().to(device)
    node_type = torch.from_numpy(node_type).long().to(device)

    mean = torch.from_numpy(train_dataset.mean).float().to(device)
    std = torch.from_numpy(train_dataset.std).float().to(device)
    max = torch.from_numpy(train_dataset.max).float().to(device)
    min = torch.from_numpy(train_dataset.min).float().to(device)
    model = TSMixer(config['in_len'], config['out_len'], input_channels=1, output_channels=1)
    model = model.to(device)

    train_process = tqdm(range(config['epochs']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    optimizer_scheduler = CosineLRScheduler(optimizer, config['epochs'], lr_min=1e-6, warmup_lr_init=1e-5, t_in_epochs=True, k_decay=1.0)

    best_mae_chla = 100

    for epoch in train_process:
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        model.train()
        optimizer_scheduler.step(epoch)
        end = time.time()
        for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
            datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.float().to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)
            B, T, C, N = datas.shape
            datas = rearrange(datas, 'b t c n->(b n) t c')
            prediction = model(datas)
            prediction = rearrange(prediction, '(b n) t c -> b t c n', b=B, n=N)
            loss = masked_huber_loss(prediction, labels, label_masks)

            loss.backward()
            optimizer.step()

            losses_m.update(loss.item(), datas.size(0))
            data_time_m.update(time.time() - end)
            torch.cuda.synchronize()

        log_buffer = "train prediction loss : {:.4f}".format(losses_m.avg)
        log_buffer += "| time : {:.4f}".format(data_time_m.avg)
        train_process.write(log_buffer)
        losses_m.reset()
        data_time_m.reset()
        if epoch % config['test_freq'] == 0 and epoch != 0:
            predictions = []
            labels_list = []
            label_masks_list = []
            datas_list = []
            data_masks_list = []
            with torch.no_grad():
                for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
                    datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.float().to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)

                    B, T, C, N = datas.shape
                    datas = rearrange(datas, 'b t c n->(b n) t c')
                    prediction = model(datas)
                    prediction = rearrange(prediction, '(b n) t c -> b t c n', b=B, n=N)
                    mask = label_masks.cpu()
                    label = labels.cpu()

                    predictions.append(prediction[:,:,0].cpu())
                    labels_list.append(label[:,:,0])
                    label_masks_list.append(mask[:,:,0])

            predictions = torch.cat(predictions, 0)
            labels = torch.cat(labels_list, 0)
            label_masks = torch.cat(label_masks_list, 0)
            chla_mae = (torch.abs(predictions - labels) * label_masks).sum([1]) / (label_masks.sum([1]) + 1e-5)
            chla_mse = (((predictions - labels) * label_masks) ** 2).sum([1]) / (label_masks.sum([1]) + 1e-5)
            chla_mae = chla_mae.mean(0)
            chla_mse = chla_mse.mean(0)
            chla_mae_mean = chla_mae.mean()
            chla_mse_mean = chla_mse.mean()
            chla_mae_area1_mean = chla_mae[train_dataset.area1.astype(bool)].mean()
            chla_mae_area2_mean = chla_mae[train_dataset.area2.astype(bool)].mean()
            chla_mae_area3_mean = chla_mae[train_dataset.area3.astype(bool)].mean()
            chla_mse_area1_mean = chla_mse[train_dataset.area1.astype(bool)].mean()
            chla_mse_area2_mean = chla_mse[train_dataset.area2.astype(bool)].mean()
            chla_mse_area3_mean = chla_mse[train_dataset.area3.astype(bool)].mean()


            log_buffer = "MAE: all area mean is {:.4f}, area1 mean is {:.4f}, area2 mean is {:.4f}, area3 mean is {:.4f}\t".format(chla_mae_mean, chla_mae_area1_mean, chla_mae_area2_mean, chla_mae_area3_mean)
            log_buffer += "MSE: all area mean is {:.4f}, area1 mean is {:.4f}, area2 mean is {:.4f}, area3 mean is {:.4f}".format(chla_mse_mean, chla_mse_area1_mean, chla_mse_area2_mean, chla_mse_area3_mean)

            train_process.write(log_buffer)
            logging.info(log_buffer)

            if chla_mae_mean < best_mae_chla:
                torch.save(model, base_dir + "best_tsmixer_with_imputation_{}.pt".format(nth_test))
                np.save(base_dir + "prediction_{}.npy".format(str(index)), predictions)
                best_mae_chla = chla_mae_mean

if __name__=="__main__":
    main(args.nth)
