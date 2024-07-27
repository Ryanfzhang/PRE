import warnings
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import h5py
import os
import pickle
from utils import StandardScaler, LogScaler
import copy
warnings.filterwarnings("ignore")

class PRE8dDataset(Dataset):
    def __init__(self, data_root, in_len, out_len, missing_ratio=0.1, mode="train", index="all"):
        super().__init__()
        self.data_root = data_root
        self.in_len = in_len
        self.out_len = out_len
        self.shape = (60, 96)

        self.datapath = (
            self.data_root + "/missing_" + str(missing_ratio) + "_in_" + str(in_len) + "_out_" + str(out_len) + "_1_imputed.pk"
        )
        self.mode = mode
        oral_data, oral_mask, mean = self.load_data()
        self.oral_data = oral_data
        self.oral_mask = oral_mask
        self.imputation = mean
        if os.path.isfile(self.datapath) is False:
            print(self.datapath + ': data is not prepared')
        else:  # load datasetfile
            with open(self.datapath, "rb") as f:
                self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = pickle.load(
                    f
                )
        if index=="all":
            self.datas = self.datas
        else:
            self.index = int(index)
            self.datas = self.datas[:,self.index]
        if mode == "train":
            bound = 648 - self.in_len - self.out_len
            self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = self.datas[:bound], self.data_ob_masks[:bound], self.data_gt_masks[:bound], self.labels[:bound], self.label_ob_masks[:bound]
        elif mode == 'test':
            bound = 648 - self.in_len - self.out_len
            self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = self.datas[bound:], self.data_ob_masks[bound:], self.data_gt_masks[bound:], self.labels[bound:], self.label_ob_masks[bound:]

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        return self.datas[index], self.data_ob_masks[index], self.data_gt_masks[index], self.labels[index], self.label_ob_masks[index]

    def load_data(self):

        data = h5py.File(osp.join(self.data_root, "modis_chla_8d_4km_pre.mat"))
        chla = np.array(data["CHLA_pre"])[:, np.newaxis]
        chla_mask = ~np.isnan(chla)
        self.chla_scaler = LogScaler()
        chla = self.chla_scaler.transform(chla)
        chla = np.nan_to_num(chla, nan=0.)

        mean_d = np.nanmean(np.reshape(chla[:960], (960//48, 48, 1, chla.shape[2], chla.shape[3])), axis=0)
        mean = np.concatenate([np.tile(mean_d, (960//48, 1, 1, 1)), mean_d[:17]], axis=0)
        mask = chla_mask.astype(np.float32)
        return chla, mask, mean

if __name__ == "__main__":
    dataset = PRE8dDataset(data_root='./data/PRE/8d', in_len=12, out_len=12,missing_ratio=0.1, mode='test', index=4)
    data, mask, _, _, _ = dataset.__getitem__(100)
    print(data)
    print(mask)