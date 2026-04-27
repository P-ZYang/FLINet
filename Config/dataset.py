import os
import torch
from PIL import Image
import numpy as np
import random
from torch.utils.data.dataset import Dataset

def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']

def random_crop(img, mask, patch_size, pos_prob=None):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                     mode='constant')  # 将不足 256的一边填充至256
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                      mode='constant')  # label 与image 进行相同的变换
        h, w = img.shape

    while 1:
        h_start = random.randint(0, h - patch_size)
        h_end = h_start + patch_size
        w_start = random.randint(0, w - patch_size)
        w_end = w_start + patch_size

        img_patch = img[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]

        if pos_prob == None or random.random() > pos_prob:
            break
        elif mask_patch.sum() > 0:
            break

    return img_patch, mask_patch

def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img

def get_img_norm_cfg(dataset_name, dataset_dir):
    norm_configs = {
        'NUAA-SIRST': dict(mean=101.06385040283203, std=34.619606018066406),
        'NUDT-SIRST': dict(mean=107.80905151367188, std=33.02274703979492),
        'IRSTD-1K': dict(mean=87.4661865234375, std=39.71953201293945),
    }
    if dataset_name in norm_configs:
        return norm_configs[dataset_name]
    else:
        # Compute normalization from actual dataset images
        with open(dataset_dir + '/' + dataset_name + '/train' + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir + '/' + dataset_name + '/test' + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        return dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        self.tranform = augumentation()
        with open(self.dataset_dir + '/train' + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()

    def __getitem__(self, idx):

        img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert('I')  # read image base on version ”I“
        # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
        mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/test' + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//', '/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        # if mask.shape == (416,608):
        #     print('111')
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape

        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        if img.size() != mask.size():
            print('111')
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:  # 水平反转
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target

