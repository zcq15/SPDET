import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import numpy as np

import random

import sys
sys.path.append('..')
from models.gargs import _args

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class ThreeD60(Dataset):
    def __init__(self,datadir,area,imgsize=(256,512),split=['s2d3d','m3d'],aug=False,local_rank=0):
        super().__init__()
        self.datadir = os.path.normpath(datadir)
        self.area = area
        self.imgsize = imgsize
        self.files = []
        assert area in ['train','val','test']
        if area == 'train':
            data_list = os.path.join(self.datadir,'train.txt')
        else:
            data_list = os.path.join(self.datadir,'test.txt')
        with open(data_list,'r') as f:
            for line in f.readlines():
                if line.strip() == '':
                    continue
                if (not 's2d3d' in split) and 'Stanford2D3D' in line:
                    continue
                if (not 'm3d' in split) and 'Matterport3D' in line:
                    continue
                rgb,right,up,depth,depth_rt,depth_up = line.strip().split(' ')[:6]
                fn = os.path.basename(line.strip().split(' ')[0])[:-26]
                self.files.append({'fn':fn+'_up','rgb':rgb,'target':up,'depth':depth,'tardep':depth_up,'pos':'up'})
                if area == 'train':
                    self.files.append({'fn':fn+'_right','rgb':rgb,'target':right,'depth':depth,'tardep':depth_rt,'pos':'right'})
                

        self.size = len(self.files)
        print('Load {} items in {} for {} split ...'.format(self.size,self.datadir,self.area)) if local_rank==0 else None

        self.aug = aug
        self.m = 1
        self.max_depth_meters = _args['data']['max']
        self.min_depth_meters = _args['data']['min']
        self.to_tensor = transforms.ToTensor()

        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1
        self.color_aug = transforms.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue)

    def __len__(self):
        return self.size

    def calc(self,rgb_src,rgb_tar,depth,tardep,pos,aug_prob=0.5):
        if self.area == 'train' and self.aug and random.random() < aug_prob:
            rgb_src_aug = np.asarray(self.color_aug(transforms.ToPILImage()(rgb_src)))
            rgb_tar_aug = np.asarray(self.color_aug(transforms.ToPILImage()(rgb_tar)))
        else:
            rgb_src_aug = rgb_src
            rgb_tar_aug = rgb_tar

        rgb_src = self.to_tensor(rgb_src.copy())
        rgb_src_aug = self.to_tensor(rgb_src_aug.copy())
        rgb_tar = self.to_tensor(rgb_tar.copy())
        rgb_tar_aug = self.to_tensor(rgb_tar_aug.copy())
        depth[depth <= self.min_depth_meters] = 0
        depth[depth >= self.max_depth_meters] = 0
        tardep[tardep <= self.min_depth_meters] = 0
        tardep[tardep >= self.max_depth_meters] = 0
        depth = torch.from_numpy(np.expand_dims(depth, axis=0)).type(torch.float32)
        tardep = torch.from_numpy(np.expand_dims(tardep, axis=0)).type(torch.float32)
        valid = ((depth > self.min_depth_meters) & (depth < self.max_depth_meters) & ~torch.isnan(depth))

        if pos == 'right':
            pos_rel = torch.tensor([0,0.26,0],dtype=torch.float32).view([3,1,1])
        else:
            pos_rel = torch.tensor([0,0,0.26],dtype=torch.float32).view([3,1,1])
        
        pos_zero = torch.zeros([3,1,1],dtype=torch.float32)

        if self.area == 'train' and self.aug and random.random() < aug_prob:
            pos_rel[1] = - pos_rel[1]
            return {'src_rgb':rgb_src_aug.flip(-1), 'source':rgb_src.flip(-1), 'src_depth':depth.flip(-1), 'src_valid':valid.flip(-1),'src_pos':pos_zero,
                    'tar_rgb':rgb_tar_aug.flip(-1), 'target':rgb_tar.flip(-1), 'tar_depth':tardep.flip(-1), 'tar_pos':pos_rel}
        else:
            return {'src_rgb':rgb_src_aug, 'source':rgb_src, 'src_depth':depth, 'src_valid':valid,'src_pos':pos_zero,
                    'tar_rgb':rgb_tar_aug, 'target':rgb_tar, 'tar_depth':tardep, 'tar_pos':pos_rel}

    def __getitem__(self,index):
        fn = self.files[index]['fn']
        rgb = self.files[index]['rgb']
        depth = self.files[index]['depth']
        target = self.files[index]['target']
        tardep = self.files[index]['tardep']
        pos = self.files[index]['pos']

        rgb = cv2.imread(os.path.join(rgb),cv2.IMREAD_COLOR)
        depth = cv2.imread(os.path.join(depth),cv2.IMREAD_ANYDEPTH)
        target = cv2.imread(os.path.join(target),cv2.IMREAD_COLOR)
        tardep = cv2.imread(os.path.join(tardep),cv2.IMREAD_ANYDEPTH)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = depth.astype(np.float32)/self.m
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        tardep = tardep.astype(np.float32)/self.m

        tensors = self.calc(rgb,target,depth,tardep,pos)
        tensors['fn'] = fn
        tensors['area'] = self.area

        return tensors

class ThreeD60M3D(ThreeD60):
    def __init__(self,datadir,area,aug=False,local_rank=0):
        super().__init__(datadir,area,split=['m3d'],aug=aug,local_rank=local_rank)

class ThreeD60S2D3D(ThreeD60):
    def __init__(self,datadir,area,aug=False,local_rank=0):
        super().__init__(datadir,area,split=['s2d3d'],aug=aug,local_rank=local_rank)