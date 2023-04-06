import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
from PIL import Image
import numpy as np

from collections import OrderedDict
import os
import random

import sys
sys.path.append('..')
from models.gargs import _args

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class PNVSDepth(Dataset):
    def __init__(self,datadir,area,imgsize=(256,512),split=['easy','hard'],aug=True,local_rank=0,**kargs):
        super().__init__()
        self.datadir = os.path.normpath(datadir)
        self.area = area
        self.imgsize = tuple(imgsize)
        self.easy_files = []
        self.hard_files = []
        self.wrong_files = []
        if area == 'train':
            self.list = 'train.txt'
        else:
            self.list = 'val.txt'

        if 'easy' in split:
            with open(os.path.join(self.datadir,'easy','wrong.txt')) as f:
                for line in f.readlines():
                    self.wrong_files.append(line.strip())
            with open(os.path.join(self.datadir,'easy',self.list)) as f:
                for line in f.readlines():
                    if line.strip() in self.wrong_files:
                        continue
                    if self.area == 'train':
                        self.easy_files.append(line.strip())
                    elif line.strip()[-1] == '0':
                        self.easy_files.append(line.strip())

        if 'hard' in split:
            with open(os.path.join(self.datadir,'hard',self.list)) as f:
                for line in f.readlines():
                    if self.area == 'train':
                        self.hard_files.append(line.strip())
                    elif line.strip()[-1] == '0':
                        self.hard_files.append(line.strip())

        self.size = len(self.easy_files)+len(self.hard_files)
        print('Load {} items in {} ...'.format(self.size,self.datadir)) if local_rank==0 else None

        self.m = 1000
        self.max_depth_meters = _args['data']['max']
        self.min_depth_meters = _args['data']['min']
        self.to_tensor = transforms.ToTensor()

        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1
        self.color_aug = transforms.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue)
        
        self.aug = aug

    def calc(self,rgb_src,depth_src,pos_src,rgb_tar,pos_tar,aug_prob=0.5,**kargs):

        if self.area == 'train' and self.aug and random.random() < aug_prob:
            rgb_src_aug = np.asarray(self.color_aug(transforms.ToPILImage()(rgb_src)))
            rgb_tar_aug = np.asarray(self.color_aug(transforms.ToPILImage()(rgb_tar)))
        else:
            rgb_src_aug = rgb_src
            rgb_tar_aug = rgb_tar

        rgb_src_aug = self.to_tensor(rgb_src_aug.copy())
        rgb_src = self.to_tensor(rgb_src.copy())

        depth_src = depth_src.astype(np.float32)/self.m
        depth_src[depth_src <= self.min_depth_meters] = 0
        depth_src[depth_src >= self.max_depth_meters] = 0
        depth_src = torch.from_numpy(np.expand_dims(depth_src, axis=0)).type(torch.float32)

        valid_src = ((depth_src > self.min_depth_meters) & (depth_src < self.max_depth_meters) & ~torch.isnan(depth_src))

        rgb_tar_aug = self.to_tensor(rgb_tar_aug.copy())
        rgb_tar = self.to_tensor(rgb_tar.copy())

        pos_zero = torch.zeros([3,1,1],dtype=torch.float32)
        pos_rel = torch.from_numpy(np.array([-(pos_tar[1]-pos_src[1]),pos_tar[0]-pos_src[0],pos_tar[2]-pos_src[2]],dtype=np.float32)/1000).view(3,1,1)

        if self.area == 'train' and self.aug and random.random() < aug_prob:
            pos_rel[1] = - pos_rel[1]
            return {'src_rgb':rgb_src_aug.flip(-1), 'source':rgb_src.flip(-1), 'src_depth':depth_src.flip(-1),'src_valid':valid_src.flip(-1),'src_pos':pos_zero,
                    'tar_rgb':rgb_tar_aug.flip(-1), 'target':rgb_tar.flip(-1), 'tar_pos':pos_rel}
        else:
            return {'src_rgb':rgb_src_aug, 'source':rgb_src, 'src_depth':depth_src, 'src_valid':valid_src,'src_pos':pos_zero,
                    'tar_rgb':rgb_tar_aug, 'target':rgb_tar, 'tar_pos':pos_rel}

        # return {'rgb':rgb_src_aug, 'source':rgb_src, 'depth':depth_src, 'valid':valid_src,
        #             'tar':rgb_tar_aug, 'target':rgb_tar, 'pos':pos_rel}


    def __len__(self):
        return self.size

    def __getitem__(self,index):
        if index < len(self.easy_files):
            split = 'easy'
            files = self.easy_files
        else:
            split = 'hard'
            files = self.hard_files
            index = index - len(self.easy_files)
        scene,room,pos = files[index].split()
        fn = '{}_{}_{}_{}'.format(split,scene,room,pos)
        source_camera = '{}_{}.txt'.format(scene,room)
        source_image = '{}_{}.png'.format(scene,room)
        source_depth = '{}_{}.png'.format(scene,room)
        target_camera = '{}_{}_{}.txt'.format(scene,room,pos)
        target_image = '{}_{}_{}.png'.format(scene,room,pos)

        pos_src = np.loadtxt(os.path.join(self.datadir,split,'source_camera',source_camera),dtype=np.float32)
        rgb_src = cv2.imread(os.path.join(self.datadir,split,'source_image',source_image),cv2.IMREAD_COLOR)
        rgb_src = cv2.cvtColor(rgb_src,cv2.COLOR_BGR2RGB)
        depth_src = cv2.imread(os.path.join(self.datadir,split,'source_depth',source_depth),-1)
        pos_tar = np.loadtxt(os.path.join(self.datadir,split,'target_camera',target_camera),dtype=np.float32)
        rgb_tar = cv2.imread(os.path.join(self.datadir,split,'target_image',target_image),cv2.IMREAD_COLOR)
        rgb_tar = cv2.cvtColor(rgb_tar,cv2.COLOR_BGR2RGB)

        assert pos_src.shape == (3,) and pos_tar.shape == (3,)
        if not tuple(rgb_src.shape[:2]) == self.imgsize:
            rgb_src = cv2.resize(rgb_src,self.imgsize[::-1],cv2.INTER_CUBIC)
            depth_src = cv2.resize(depth_src,self.imgsize[::-1],cv2.INTER_NEAREST)
            rgb_tar = cv2.resize(rgb_tar,self.imgsize[::-1],cv2.INTER_CUBIC)

        tensors = self.calc(rgb_src,depth_src,pos_src,rgb_tar,pos_tar)
        tensors['area'] = self.area
        tensors['fn'] = fn
        return tensors

class EasyPNVSDepth(PNVSDepth):
    def __init__(self,datadir,area,imgsize=(256,512),aug=True,local_rank=0):
        super().__init__(datadir=datadir,area=area,split=['easy'],imgsize=imgsize,aug=aug,local_rank=local_rank)

class HardPNVSDepth(PNVSDepth):
    def __init__(self,datadir,area,imgsize=(256,512),aug=True,local_rank=0):
        super().__init__(datadir=datadir,area=area,split=['hard'],imgsize=imgsize,aug=aug,local_rank=local_rank)
