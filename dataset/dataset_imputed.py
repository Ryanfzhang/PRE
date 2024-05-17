import warnings
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import h5py
import os
import pickle
import torch
import copy
from utils import StandardScaler, LogScaler
warnings.filterwarnings("ignore")


class PREDataset(Dataset):
    def __init__(self, data_root, in_len, out_len, missing_ratio=0.1, mode="train"):
        super().__init__()
        self.data_root = data_root
        self.in_len = in_len
        self.out_len = out_len
        self.shape = (60, 96)

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.datapath = (
            self.data_root + "/missing_" + str(missing_ratio) + "_in_" + str(in_len) + "_out_" + str(out_len) + "_imputed.pk"
        )
        self.mode = mode
        oral_data, oral_mask, mean = self.load_data()
        mask_in_ocean = copy.deepcopy(oral_mask)
        index = np.sum(mask_in_ocean, 0) == 0
        mask_in_ocean[:, index] = 1.

        self.oral_data = oral_data
        self.oral_mask = oral_mask
        self.imputation = mean

        imputer = torch.load("/home/mafzhang/code/Project/PRE/PRE_CHLA_DIFFUSION/runs/best.pt")
        imputer = imputer.to(device)
        if os.path.isfile(self.datapath) is False:
            datas, data_ob_masks, data_gt_masks, labels, label_ob_masks = [], [], [], [], []
            for index in range(len(oral_data) - in_len - out_len):
                data = oral_data[index:index+self.in_len]
                data_ob_mask = oral_mask[index:index+self.in_len]
                label = oral_data[index+self.in_len:index+self.in_len+self.out_len]
                label_ob_mask = oral_mask[index+self.in_len:index+self.in_len+self.out_len]

                torch_data = torch.from_numpy(data).float().to(device)
                torch_ob_mask = torch.from_numpy(data_ob_mask).float().to(device)
                imputed_data = imputer.impute(torch_data.unsqueeze(0), torch_ob_mask.unsqueeze(0), 20)
                imputed_data = imputed_data.median(dim=1).values

                data = np.where(mask_in_ocean[index:index+self.in_len].astype(bool), data, imputed_data.numpy().squeeze(0))

                masks = data_ob_mask.reshape(-1).copy()
                obs_indices = np.where(masks)[0].tolist()
                miss_indices = np.random.choice(
                    obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
                )
                masks[miss_indices] = False
                gt_masks = masks.reshape(data_ob_mask.shape)
                datas.append(data)
                data_ob_masks.append(data_ob_mask)
                data_gt_masks.append(gt_masks)
                labels.append(label)
                label_ob_masks.append(label_ob_mask)

            self.datas = np.array(datas).astype("float32")
            self.data_ob_masks = np.array(data_ob_masks).astype("float32")
            self.data_gt_masks = np.array(data_gt_masks).astype("float32")
            self.labels = np.array(labels).astype("float32")
            self.label_ob_masks = np.array(label_ob_masks).astype("float32")

            with open(self.datapath, "wb") as f:
                pickle.dump(
                    [self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks], f
                )
        else:  # load datasetfile
            with open(self.datapath, "rb") as f:
                self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = pickle.load(
                    f
                )
        if mode == "train":
            bound = 168 - self.in_len - self.out_len
            self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = self.datas[:bound], self.data_ob_masks[:bound], self.data_gt_masks[:bound], self.labels[:bound], self.label_ob_masks[:bound]
        elif mode == 'test':
            bound = 168 - self.in_len - self.out_len
            self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = self.datas[bound:], self.data_ob_masks[bound:], self.data_gt_masks[bound:], self.labels[bound:], self.label_ob_masks[bound:]

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        return self.datas[index], self.data_ob_masks[index], self.data_gt_masks[index], self.labels[index], self.label_ob_masks[index]

    def load_data(self):
        data = h5py.File(osp.join(self.data_root, "modis_sst_month_4km_pre.mat"))
        sst = np.array(data["SST_pre"])
        self.sst_scaler = StandardScaler()
        sst_mask = ~np.isnan(sst)
        sst = self.sst_scaler.transform(sst)
        sst = np.nan_to_num(sst, nan=0.)

        data = h5py.File(osp.join(self.data_root, "modis_chla_month_4km_pre.mat"))
        chla = np.array(data["CHLA_pre"])
        chla_mask = ~np.isnan(chla)
        self.chla_scaler = LogScaler()
        chla = self.chla_scaler.transform(chla)
        chla = np.nan_to_num(chla, nan=0.)

        data = np.stack([sst, chla], axis=1)
        mask = np.stack([sst_mask, chla_mask], axis=1)
        mean = np.nanmean(np.reshape(data, (228//12, 12, 2, data.shape[2], data.shape[3])), axis=0)
        mean = np.tile(mean, (228//12, 1, 1, 1))
        mask = mask.astype(np.float32)
        return data, mask, mean


if __name__ == "__main__":
    dataset = PREDataset(data_root='./data/PRE', in_len=6, out_len=1, mode='train')
    print(dataset.datas.shape)
