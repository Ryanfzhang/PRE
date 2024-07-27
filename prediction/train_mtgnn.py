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
import argparse

from dataset_pre.dataset_for_graph_imputed import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_huber_loss
from model.mtgnn import MTGNN

parser = argparse.ArgumentParser()
parser.add_argument('--nth', default=0)
args = parser.parse_args()

def main(nth_test):
    with open("./config/config_prediction.yaml", 'r') as f:
        config = yaml.safe_load(f)

    base_dir = "./log_prediction/mtgnn/input_{}_output_{}".format(str(config['in_len']), str(config['out_len']))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    check_dir(base_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, 'train_prediction_{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
    print(config)
    logging.info(config)


    train_dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='train', index=nth_test)
    train_dloader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    test_dloader = DataLoader(PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d/', in_len=config['in_len'], out_len=config['out_len'], missing_ratio=config['missing_ratio'], mode='test', index=nth_test), 1, shuffle=False)
    adj = train_dataset.adj
    adj = torch.from_numpy(adj).float().to(device)
    model = MTGNN(gcn_true=True, build_adj=False, kernel_set=[7,7], kernel_size=7, gcn_depth=1, num_nodes=4443, dropout=0.3, subgraph_size=20, node_dim=8,dilation_exponential=2, conv_channels=2, residual_channels=2, skip_channels=4, end_channels=8, seq_length=config['in_len'], in_dim=1, out_dim=config['out_len'], layers=2, propalpha=0.5, tanhalpha=3, layer_norm_affline=False)
    model = model.to(device)
    index = str(nth_test)

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
            datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.float().to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)

            prediction = model(datas, adj)
            loss = masked_huber_loss(prediction, labels, label_masks)
            losses_m.update(loss.item(), datas.size(0))
            data_time_m.update(time.time() - end)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
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
            with torch.no_grad():
                for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
                    datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.float().to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)

                    prediction = model(datas, adj)
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
                torch.save(model, base_dir + "best_mtgnn_with_imputation_{}.pt".format(nth_test))
                np.save(base_dir + "prediction_{}.npy".format(str(index)), predictions)
                best_mae_chla = chla_mae_mean

if __name__=="__main__":
    main(args.nth)
