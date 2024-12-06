from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            # Nazish: Add a piece of code here to iteratively read HR and SR for different bands and concatnate to the original
            img_HR = np.array(Image.open(self.hr_path[index]).convert("RGB"))
            img_SR = np.array(Image.open(self.sr_path[index]).convert("RGB"))
            for i in range(4,12,1):
                if i!=9:
                    #fix the path name
                    hr_path = self.hr_path[index].split('/')
                    # print(hr_path)
                    hr_path[-3]="B"+str(i)
                    hr_path = "/".join(hr_path)

                    sr_path = self.sr_path[index].split('/')
                    sr_path[-3]="B"+str(i)
                    sr_path = "/".join(sr_path)

                    img_HR_b = np.array(Image.open(self.hr_path[index]).convert("RGB"))
                    img_SR_b = np.array(Image.open(self.sr_path[index]).convert("RGB"))

                    img_HR = np.concatenate((img_HR, np.expand_dims(img_HR_b[:,:,0],axis = 2)), axis=2)
                    img_SR = np.concatenate((img_SR, np.expand_dims(img_SR_b[:,:,0],axis = 2)), axis=2)




            if self.need_LR:
                img_LR = np.array(Image.open(self.lr_path[index]).convert("RGB"))

                for i in range(4,12,1):
                    if i!=9:
                        lr_path = self.lr_path[index].split('/')
                        lr_path[-3]="B"+str(i)
                        lr_path = "/".join(lr_path)
                        img_LR_b = np.array(Image.open(self.lr_path[index]).convert("RGB"))
                        img_LR = np.concatenate((img_LR, np.expand_dims(img_LR_b[:,:,0],axis = 2)), axis=2)
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
