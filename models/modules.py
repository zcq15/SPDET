import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('.')
from .gargs import _args

class ZeroPad(nn.Module):
    def __init__(self, padding):
        super(ZeroPad, self).__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]
    
    def forward(self, x):
        x = F.pad(x, (self.w, self.w, self.h, self.h)) 
        return x

class CircPad(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]

    def forward(self,x):
        _,_,H,W = x.shape
        assert H%2==0 and W%2==0
        if self.h == 0 and self.w == 0:
            return x
        elif self.h == 0:
            return F.pad(x,pad=(self.w,self.w,0,0),mode='circular')
        else:
            up = x[:,:,:self.h].flip(2).roll(W//2,dims=-1)
            down = x[:,:,-self.h:].flip(2).roll(W//2,dims=-1)
            return F.pad(torch.cat([up,x,down],dim=2),pad=(self.w,self.w,0,0),mode='circular')

class LRPad(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]

    def forward(self,x):
        _,_,H,W = x.shape
        assert H%2==0 and W%2==0
        if self.h==0 and self.w==0:
            return x
        return F.pad(F.pad(x,pad=(self.w,self.w,0,0),mode='circular'),pad=(0,0,self.h,self.h))

def _make_pad(padding=0,pad=None,**kargs):
    if pad is None and 'padding' in _args['model']:
        pad = _args['model']['padding']
    if pad == 'circpad':
        return CircPad(padding)
    elif pad == 'lrpad':
        return LRPad(padding)
    elif pad == 'zeropad':
        return ZeroPad(padding)
    else:
        return CircPad(padding)

def _make_norm(norm,layers,**kargs):
    if norm is None or norm == 'idt' or norm == 'none':
        return nn.Identity()
    elif norm == 'bn':
        return nn.BatchNorm2d(layers)
    elif norm == 'inst':
        return nn.InstanceNorm2d(layers)
    elif norm == 'gn':
        if not 'groups' in kargs:
            return nn.GroupNorm(32,layers)
        else:
            return nn.GroupNorm(kargs['groups'],layers)
    else:
        raise NotImplementedError

def _make_act(act,**kargs):
    if act is None or act == 'idt':
        return nn.Identity()
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.01,inplace=True)
    elif act == 'orelu':
        return nn.ReLU(inplace=False)
    elif act == 'olrelu':
        return nn.LeakyReLU(negative_slope=0.01,inplace=False)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError

def spherical_upsample(x,scale):
    _,_,h,w = x.shape
    assert h%2==0 and w%2==0
    up = x[:,:,:1].flip(2).roll(w//2,dims=-1)
    down = x[:,:,-1:].flip(2).roll(w//2,dims=-1)
    x = F.pad(torch.cat([up,x,down],dim=2),pad=(1,1,0,0),mode='circular')
    x = F.interpolate(x,scale_factor=scale,mode='bilinear',align_corners=False)
    x = x[:,:,scale:-scale,scale:-scale]
    return x

def spherical_downsample(x,scale):
    _,_,h,w = x.shape
    s = scale
    assert h%2==0 and w%2==0
    up = x[:,:,:s].flip(2).roll(w//2,dims=-1)
    down = x[:,:,-s:].flip(2).roll(w//2,dims=-1)
    x = F.pad(torch.cat([up,x,down],dim=2),pad=(s,s,0,0),mode='circular')
    x = F.interpolate(x,scale_factor=1/s,mode='bilinear',align_corners=False)
    x = x[:,:,1:-1,1:-1]
    return x

def spherical_reminder(grid,size,inplace=False,indexing='ij'):
    assert grid.shape[-1] == 2 and len(size) == 2
    assert size[0]%2 == 0 and size[1]%2 == 0
    grid = torch.where(torch.isinf(grid) | torch.isnan(grid), torch.zeros_like(grid), grid)
    if indexing == 'ij':
        grid_h = grid[...,0]
        grid_w = grid[...,1]
    else:
        grid_h = grid[...,1]
        grid_w = grid[...,0]
    h = size[0]
    w = size[1]

    while (grid_h.min() < 0 or grid_h.max() > h-1 or grid_w.min() < 0 or grid_w.max() > w):
        if inplace:
            overflow = grid_h<0
            grid_h[overflow] = -grid_h[overflow]
            grid_w[overflow] = grid_w[overflow]+w//2
            underflow = grid_h>(h-1)
            grid_h[underflow] = 2*(h-1)-grid_h[underflow]
            grid_w[underflow] = grid_w[underflow]+w//2
            grid_w = torch.remainder(grid_w,w)
        else:
            overflow = grid_h<0
            grid_h = torch.where(overflow,-grid_h,grid_h)
            grid_w = torch.where(overflow,grid_w+w//2,grid_w)
            underflow = grid_h>(h-1)
            grid_h = torch.where(underflow,2*(h-1)-grid_h,grid_h)
            grid_w = torch.where(underflow,grid_w+w//2,grid_w)
            grid_w = torch.remainder(grid_w,w)
    if indexing == 'ij':
        grid = torch.stack([grid_h,grid_w],dim=-1)
    else:
        grid = torch.stack([grid_w,grid_h],dim=-1)
    return grid

def spherical_grid_sample(x,grid,clip=False,inplace=False,indexing='ij'):
    B,C,H,W = x.shape
    assert len(grid.shape) == 4 and grid.shape[-1] == 2
    # assert tuple(grid.shape) == (B,H,W,2)

    if clip:
        grid = spherical_reminder(grid,[H,W],inplace=inplace,indexing=indexing)
    if indexing == 'ij':
        grid_h = grid[...,0]
        grid_w = grid[...,1]
    else:
        grid_h = grid[...,1]
        grid_w = grid[...,0]

    x = torch.cat([x,x[...,:1]],dim=-1)
    grid = torch.stack([grid_w/W*2-1,grid_h/(H-1)*2-1],dim=-1)
    y = torch.nn.functional.grid_sample(x,grid,align_corners=True)
    return y