from models import simvp, mtgnn, tcu,tau
from metric import metric
from dataset import CHLADataset
from main_utils import check_dir, gather_tensors_batch,masked_mae_loss,masked_huber_loss, masked_mape_loss
from cross_models import cross_former

import os
import time
import argparse
import torch
import logging
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np

parser = argparse.ArgumentParser(description='CHLA Prediction Official Implement')
# Dataset and Model Parameters parser.add_argument('-bp', '--base-path', default="./")
parser.add_argument('--method', default="simvp")
parser.add_argument('--type', default="tau")
parser.add_argument('-sl', '--sequence_length', default=6, type=int)
parser.add_argument('--seqlen_out', default=1, type=int)
parser.add_argument('-cin', '--channel_in', default=2, type=int)
parser.add_argument('-cout', '--channel_out', default=1, type=int)
# Train Strategy Parameters
parser.add_argument('-bs', '--batch_size', default=1, type=int)
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=int)
parser.add_argument('-se', '--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-ee', '--end_epoch', default=300, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-pf', '--print_freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
# Optimizer Parameters
parser.add_argument('--optimizer', default="Adam", type=str, metavar="Optimizer Name")
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float)
parser.add_argument("--gpu", default="0", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
args = parser.parse_args()
def main(args):

    torch.manual_seed(42)
    device = torch.device('cuda:0')

    # if args.method=="tcu":
    #     model = tcu.TCU(seqlen_in=args.sequence_length, seqlen_out=args.sequence_length, channel_in=args.channel_in, channel_out=args.channel_out, H=60, W=96)
    #     model = model.to(device)
    # elif args.method=="tau":
    #     model = tau.TAU(seqlen_in=args.sequence_length, seqlen_out=args.sequence_length, channel_in=args.channel_in, channel_out=args.channel_out, H=60, W=96)
    #     model = model.to(device)
    # elif args.method =="mtgnn":
    #     model = mtgnn.MTGNN(gcn_true=True, build_adj=False, kernel_set=[7,7], kernel_size=7, gcn_depth=1, num_nodes=60*96, dropout=0.3, subgraph_size=20, node_dim=8,dilation_exponential=2, conv_channels=2, residual_channels=2, skip_channels=4, end_channels=8, seq_length=args.sequence_length, in_dim=2, out_dim=args.sequence_length, layers=2, propalpha=0.5, tanhalpha=3, layer_norm_affline=False)
    #     model = model.to(device)
    # elif args.method =="crossformer":
    #     model = cross_former.Crossformer(args.channel_in, args.sequence_length, args.sequence_length, 6)
    #     model = model.to(device)
    #     args.batch_size = 1
    # else:
    #     print("Not Implement")
    model = torch.nn.LSTM(args.channel_in, args.channel_out, batch_first=True)
    model = model.to(device)
    
    criterion = torch.nn.MSELoss()
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay, momentum=0.8)
        
    optimizer_scheduler = CosineLRScheduler(optimizer, args.end_epoch -args.start_epoch, lr_min=1e-6, warmup_lr_init=1e-5, t_in_epochs=True, k_decay=1.0)
    
    #Data Prepare
    train_dataset = CHLADataset("./data/CHLA/", args.sequence_length, args.sequence_length, mode="train")
    test_dataset = CHLADataset("./data/CHLA/", args.sequence_length, args.sequence_length, mode="test")
    train_dloader = DataLoader(train_dataset, args.batch_size, shuffle=False)
    test_dloader = DataLoader(test_dataset, args.batch_size, shuffle=False)


    #Path
    base_dir = "./runs/" + str(args.method)+"-"+str(args.type)
    check_dir(base_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO,
                                filename=os.path.join(base_dir, '{}.log'.format(timestamp)),
                                filemode='a', format='%(asctime)s - %(message)s')

    train_process = tqdm(range(args.start_epoch, args.end_epoch))

    best_mape, best_mae, best_rmse, best_cor = 100, 100, 100, 0

    for epoch in train_process:
        data_time_m = AverageMeter() 
        losses_m = AverageMeter()
        model.train()
        optimizer_scheduler.step(epoch)
        end = time.time()
        for train_step, (datas, labels, masks) in enumerate(train_dloader):
            datas , labels, masks = datas.to(device), labels.to(device), masks.to(device)
            datas = torch.permute(datas, (0,3,4,1,2))
            labels = torch.permute(labels, (0,3,4,1,2))
            masks = torch.permute(masks, (0,3,4,1,2))
            datas = datas.reshape(-1, args.sequence_length, args.channel_in)
            labels = labels.reshape(-1, args.sequence_length, args.channel_out)
            masks = masks.reshape(-1, args.sequence_length, args.channel_out)

            prediction,_= model(datas)
            loss = masked_huber_loss(prediction, labels, masks)
            losses_m.update(loss.item(), datas.size(0))
            data_time_m.update(time.time() - end)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            torch.cuda.synchronize()

            log_buffer = "train loss : {:.4f}".format(loss.item())
            log_buffer +=  "| time : {:.4f}".format(data_time_m.avg)
            end = time.time()
            train_process.set_description(log_buffer)

        if epoch % args.print_freq ==0:
            #Test
            data_list, prediction_list, label_list = [], [],[]
            for test_steps, (datas, labels, masks) in enumerate(test_dloader):
                datas , labels, masks = datas.to(device), labels.to(device), masks.to(device)
                b, t, c, h, w = datas.shape
                datas = torch.permute(datas, (0,3,4,1,2))
                labels = torch.permute(labels, (0,3,4,1,2))
                masks = torch.permute(masks, (0,3,4,1,2))
                datas = datas.reshape(-1, args.sequence_length, args.channel_in)
                labels = labels.reshape(-1, args.sequence_length, args.channel_out)
                masks = masks.reshape(-1, args.sequence_length, args.channel_out)
                with torch.no_grad():
                    prediction,_ = model(datas)
                
                datas = datas.reshape(b,h,w,t,c)
                labels = labels.reshape(b,h,w,t,1)
                masks = masks.reshape(b,h,w,t,1)
                prediction = prediction.reshape(b,h,w,t,1)
                datas = torch.permute(datas, (0,3,4,1,2))
                labels = torch.permute(labels, (0,3,4,1,2))
                prediction= torch.permute(prediction, (0,3,4,1,2))
                masks = torch.permute(masks, (0,3,4,1,2))

                data_list.append(datas.cpu().numpy())
                prediction_list.append(prediction.detach().cpu().numpy())
                label_list.append(labels.detach().cpu().numpy())
                torch.cuda.empty_cache()
            
            datas = np.concatenate(data_list, axis=0)
            predictions = np.concatenate(prediction_list, axis=0)
            labels = np.concatenate(label_list, axis=0)
            labels = labels

            MAE, RMSE, MAPE, r, tmp1s, tmp2s = [],[],[],[],[],[]

            for location in train_dataset.loc:
                r_tmp, MAE_tmp, RMSE_tmp, MAPE_tmp=[],[],[],[]
                for b in range(predictions.shape[0]):
                    i,j = location[0], location[1]
                    tmp1 = labels[b,:,0,i,j]
                    tmp2 = predictions[b,:,0,i,j]
                    tmp = tmp1 - tmp2
                    r_tmp.append(np.corrcoef(tmp1, tmp2)[0][1])
                    MAE_tmp.append(np.sum(np.abs(tmp))/len(tmp))
                    RMSE_tmp.append(np.sqrt(np.dot(tmp, tmp)/len(tmp)))
                    MAPE_tmp.append(np.sum(np.abs(tmp)/np.abs(tmp1))/len(tmp1))

                r.append(np.mean(r_tmp))
                MAE.append(np.mean(MAE_tmp))
                RMSE.append(np.mean(RMSE_tmp))
                MAPE.append(np.mean(MAPE_tmp))

            eval_log = "overall performance: MAE:{:.6f}\t RMSE:{:.6f}\t MAPE:{:.6f}\t R:{:.6f}".format(np.mean(MAE), np.mean(RMSE), np.mean(MAPE), np.nanmean(r))
            print(eval_log)
            logging.info(eval_log)
            

            if (int(best_mae>np.mean(MAE)) + int(best_rmse>np.mean(RMSE)) + int(best_mape>np.mean(MAPE)))>=2:
                best_rmse= np.mean(RMSE)
                best_mae = np.mean(MAE)
                best_mape= np.mean(MAPE)
                best_cor = np.mean(r)
                torch.save(model, os.path.join(base_dir, "best.pth"))

    model = torch.load(os.path.join(base_dir, "best.pth"))
    model.eval()
    data_list, prediction_list, label_list = [], [],[]
    for test_steps, (datas, labels, masks) in enumerate(test_dloader):
        datas , labels, masks = datas.to(device), labels.to(device), masks.to(device)
        b, t, c, h, w = datas.shape
        datas = torch.permute(datas, (0,3,4,1,2))
        labels = torch.permute(labels, (0,3,4,1,2))
        masks = torch.permute(masks, (0,3,4,1,2))
        datas = datas.reshape(-1, args.sequence_length, args.channel_in)
        labels = labels.reshape(-1, args.sequence_length, args.channel_out)
        masks = masks.reshape(-1, args.sequence_length, args.channel_out)
        with torch.no_grad():
            prediction,_ = model(datas)

        datas = datas.reshape(b,h,w,t,c)
        labels = labels.reshape(b,h,w,t,1)
        masks = masks.reshape(b,h,w,t,1)
        prediction = prediction.reshape(b,h,w,t,1)
        datas = torch.permute(datas, (0,3,4,1,2))
        labels = torch.permute(labels, (0,3,4,1,2))
        masks = torch.permute(masks, (0,3,4,1,2))
        prediction= torch.permute(prediction, (0,3,4,1,2))
        data_list.append(datas.cpu().numpy())
        prediction_list.append(prediction.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())
        torch.cuda.empty_cache()
    
    datas = np.concatenate(data_list, axis=0)
    predictions = np.concatenate(prediction_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    predictions = predictions
    labels = labels
    folder_path = os.path.join(base_dir, 'train_saved')
    check_dir(folder_path)
    np.save(os.path.join(folder_path, 'inputs.npy'), datas)
    np.save(os.path.join(folder_path, 'preds.npy'), 10**predictions)
    np.save(os.path.join(folder_path, 'trues.npy'), 10**labels)

    MAE, RMSE, MAPE, r, tmp1s, tmp2s = [],[],[],[],[],[]

    for location in train_dataset.loc:
        r_tmp, MAE_tmp, RMSE_tmp, MAPE_tmp=[],[],[],[]
        for b in range(predictions.shape[0]):
            i,j = location[0], location[1]
            tmp1 = labels[b,:,0,i,j]
            tmp2 = predictions[b,:,0,i,j]
            tmp = tmp1 - tmp2
            r_tmp.append(np.corrcoef(tmp1, tmp2)[0][1])
            MAE_tmp.append(np.sum(np.abs(tmp))/len(tmp))
            RMSE_tmp.append(np.sqrt(np.dot(tmp, tmp)/len(tmp)))
            MAPE_tmp.append(np.sum(np.abs(tmp)/np.abs(tmp1))/len(tmp1))

        r.append(np.mean(r_tmp))
        MAE.append(np.mean(MAE_tmp))
        RMSE.append(np.mean(RMSE_tmp))
        MAPE.append(np.mean(MAPE_tmp))

    eval_log = "overall performance: MAE:{:.6f}\t RMSE:{:.6f}\t MAPE:{:.6f}\t R:{:.6f}".format(np.mean(MAE), np.mean(RMSE), np.mean(MAPE), np.nanmean(r))
    print(eval_log)
    logging.info(eval_log)
        
if __name__=="__main__":
    main(args=args)