import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import sys
sys.path.append('.')
sys.path.append('..')
from ..modules import _make_pad
from ..geometry.geometry import spattn
from .grid_thirdparty import create_image_grid

class SPL1(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max
        self.weights = None

    def forward(self,_output,_input):
        if not self.target in _input or not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']
        else:
            mask = torch.ones_like(_input['src_depth'],dtype=torch.bool,device=_input['src_depth'].device)

        B,C,H,W = _input['src_rgb'].shape
        if self.weights is None: self.weight = spattn(H,W,'z').view(1,H,W)
        loss = 0
        for b in range(B):
            loss += torch.sum((pred[b]-target[b]).abs()*self.weight*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum()
        return loss/B

class SPMSE(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max
        self.weights = None

    def forward(self,_output,_input):
        if not self.target in _input or not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']
        else:
            mask = torch.ones_like(_input['src_depth'],dtype=torch.bool,device=_input['src_depth'].device)

        B,C,H,W = _input['src_rgb'].shape
        if self.weights is None: self.weight = spattn(H,W,'z').view(1,H,W)
        loss = 0
        for b in range(B):
            loss += torch.sum((pred[b]-target[b]).pow(2)*self.weight*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum()
        return loss/B

class SPAbsRel(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max
        self.weights = None

    def forward(self,_output,_input):
        if not self.target in _input or not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']*(target>0)
        else:
            mask = target>0

        B,C,H,W = _input['src_rgb'].shape
        if self.weights is None: self.weight = spattn(H,W,'z').view(1,H,W)
        loss = 0
        for b in range(B):
            target[b][~mask[b]] = 1.0
            loss += torch.sum((pred[b]-target[b]).abs()/target[b]*self.weight*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum()
        return loss/B

class SPSqRel(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max
        self.weights = None

    def forward(self,_output,_input):
        if not self.target in _input or not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']*(target>0)
        else:
            mask = target>0

        B,C,H,W = _input['src_rgb'].shape
        if self.weights is None: self.weight = spattn(H,W,'z').view(1,H,W)
        loss = 0
        for b in range(B):
            target[b][~mask[b]] = 1.0
            loss += torch.sum((pred[b]-target[b]).pow(2)/target[b]*self.weight*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum()
        return loss/B

class SPRMSE(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max
        self.weights = None

    def forward(self,_output,_input):
        if not self.target in _input or not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']
        else:
            mask = torch.ones_like(_input['src_depth'],dtype=torch.bool,device=_input['src_depth'].device)

        B,C,H,W = _input['src_rgb'].shape
        if self.weights is None: self.weight = spattn(H,W,'z').view(1,H,W)
        loss = 0
        for b in range(B):
            loss += torch.sqrt(torch.sum((pred[b]-target[b]).pow(2)*self.weight*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum())
        return loss/B

class SPLogERMSE(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',eps=1e-5,min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.eps = eps
        assert self.eps > 0
        self.min = min
        self.max = max
        self.weights = None

    def forward(self,_output,_input):
        if not self.target in _input or not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']*(pred>0)*(target>0)
        else:
            mask = (pred>0)*(target>0)

        B,C,H,W = _input['src_rgb'].shape
        if self.weights is None: self.weight = spattn(H,W,'z').view(1,H,W)
        loss = 0
        for b in range(B):
           delta = (torch.log(pred[b]) - torch.log(target[b])).pow(2)
           delta[~mask[b]] = 0
           loss += torch.sqrt( torch.sum(delta*self.weight*mask[b].type(torch.float32)) / mask[b].type(torch.float32).sum() )
        return loss/B

class SPLog10RMSE(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',eps=1e-5,min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.eps = eps
        assert self.eps > 0
        self.min = min
        self.max = max
        self.weights = None

    def forward(self,_output,_input):
        if not self.target in _input or not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']*(pred>0)*(target>0)
        else:
            mask = (pred>0)*(target>0)

        B,C,H,W = _input['src_rgb'].shape
        if self.weights is None: self.weight = spattn(H,W,'z').view(1,H,W)
        loss = 0
        for b in range(B):
           delta = (torch.log10(pred[b]) - torch.log10(target[b])).pow(2)
           delta[~mask[b]] = 0
           loss += torch.sqrt( torch.sum(delta*self.weight*mask[b].type(torch.float32)) / mask[b].type(torch.float32).sum() )
        return loss/B


class SPDelta(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',th=1.25,min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.th = th
        self.min = min
        self.max = max
        self.sampling = None

    def spiral_sampling(self, grid, percentage):
        b, c, h, w = grid.size()
        N = torch.tensor(h*w*percentage).int()
        sampling = torch.zeros_like(grid)[:, 0, :, :].unsqueeze(1)
        phi_k = torch.tensor(0.0).float()
        for k in torch.arange(N - 1):
            k = k.float() + 1.0
            h_k = -1 + 2 * (k - 1) / (N - 1)
            theta_k = torch.acos(h_k)
            phi_k = phi_k + torch.tensor(3.6).float() / torch.sqrt(N) / torch.sqrt(1 - h_k * h_k) \
                if k > 1.0 else torch.tensor(0.0).float()
            phi_k = torch.fmod(phi_k, 2 * np.pi)
            sampling[:, :, int(theta_k / np.pi * h) - 1, int(phi_k / np.pi / 2 * w) - 1] += 1.0
        return (sampling > 0).float()

    def forward(self,_output,_input):
        if not self.target in _input or not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)+1e-5
        target = _input[self.target].clip(self.min,self.max)+1e-5
        if 'src_valid' in _input:
            mask = _input['src_valid']*(pred>0)*(target>0)
        else:
            mask = (pred>0)*(target>0)

        B,C,H,W = _input['src_rgb'].shape
        
        if self.sampling is None: self.sampling = self.spiral_sampling(create_image_grid(H,W),0.25).view(1,H,W).cuda()
        loss = 0
        for b in range(B):
            thresh = torch.max(target[b]/pred[b], pred[b]/target[b])
            thresh[(~mask[b]) | (self.sampling < 0.5)] = self.th + 1.0
            mask_sum = torch.sum(self.sampling * mask[b].type(torch.float32))
            loss += (thresh<self.th).type(torch.float32).sum()/mask_sum
        return loss/B
