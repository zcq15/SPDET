import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('.')
from .geometry import xy_grid,uv_grid,xyz_grid

class ZeroPad(nn.Module):
    def __init__(self, padding):
        super(ZeroPad, self).__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]
    
    def forward_2d(self, x):
        x = F.pad(x, (self.w, self.w, self.h, self.h)) 
        return x

    def forward_3d(self, x):
        assert self.h == self.w
        d = self.h
        x = F.pad(x, (d, d, d, d, d, d)) 
        return x

    def forward(self,x):
        if len(x.shape) == 4:
            return self.forward_2d(x)
        elif len(x.shape) == 5:
            return self.forward_3d(x)
        else:
            exit(-1)

class CircPad(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]

    def forward_2d(self,x):
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

    def forward_3d(self,x):
        B,C,D,H,W = x.shape
        assert self.h == self.w
        d = self.h
        zeros = torch.zeros([B,C,d,H,W],dtype=torch.float32,device=x.device)
        x = torch.cat([zeros,x,zeros],dim=2)
        x = x.view(B,C*(D+2*d),H,W)
        assert H%2==0 and W%2==0
        if d == 0:
            return x.view(B,C,D+2*d,H+2*d,W+2*d)
        else:
            idx = torch.arange(-W//2,W//2,device=x.device)
            up = x[:,:,:d,idx].flip(2)
            down = x[:,:,-d:,idx].flip(2)
            return F.pad(torch.cat([up,x,down],dim=2),pad=(d,d,0,0),mode='circular').view(B,C,D+2*d,H+2*d,W+2*d)

    def forward(self,x):
        if len(x.shape) == 4:
            return self.forward_2d(x)
        elif len(x.shape) == 5:
            return self.forward_3d(x)
        else:
            exit(-1)

class LRPad(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]

    def forward_2d(self,x):
        _,_,H,W = x.shape
        assert H%2==0 and W%2==0
        if self.h==0 and self.w==0:
            return x
        return F.pad(F.pad(x,pad=(self.w,self.w,0,0),mode='circular'),pad=(0,0,self.h,self.h))

    def forward_3d(self,x):
        B,C,D,H,W = x.shape
        assert self.h == self.w
        d = self.h
        zeros = torch.zeros([B,C,d,H,W],dtype=torch.float32,device=x.device)
        x = torch.cat([zeros,x,zeros],dim=2)
        x = x.view(B,C*(D+2*d),H,W)
        assert H%2==0 and W%2==0
        if self.h==0 and self.w==0:
            return x.view(B,C,D+2*d,H+2*d,W+2*d)
        return F.pad(F.pad(x,pad=(d,d,0,0),mode='circular'),pad=(0,0,d,d)).view(B,C,D+2*d,H+2*d,W+2*d)

    def forward(self,x):
        if len(x.shape) == 4:
            return self.forward_2d(x)
        elif len(x.shape) == 5:
            return self.forward_3d(x)
        else:
            exit(-1)

def _make_pad(padding=0,pad='circpad',**kargs):
    if pad == 'circpad' or pad is None:
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

class Upsample(nn.Module):
    def __init__(self,in_channels, out_channels=None, scale=2, norm=None, act='gelu', mode="bilinear", align_corners=False, force=False):
        super().__init__()
        out_channels = out_channels or in_channels // 2
        self.scale = scale
        self.pad = _make_pad(1)
        self.up = nn.Upsample(scale_factor=scale, mode=mode, align_corners=align_corners)
        if in_channels == out_channels and not force:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Sequential(
                # _make_pad(1),
                nn.Conv2d(in_channels,out_channels,1,bias=not norm),
                _make_norm(norm,out_channels),
                _make_act(act)
            )
        
    def forward(self,x):
        x = self.pad(x)
        x = self.up(x)
        s = self.scale
        x = x[:,:,s:-s,s:-s]
        y = self.conv(x)
        return y

class OFRS(nn.Module):
    def __init__(self, in_channels, out_channels=None, scale=2, ofrs_grid='geo', norm=None, act='gelu', force=False):
        super().__init__()
        out_channels = out_channels or in_channels // 2
        mid_channels = 32
        self.scale = scale
        self.ofrs_grid_type = ofrs_grid
        self.ofrs_grid = None
        self.xy_grid = None
        self.pad = _make_pad(1)

        if self.ofrs_grid_type in ['cos','sin','xyz']:
            grid_channels = 3
        elif self.ofrs_grid_type in ['uv']:
            grid_channels = 2
        elif self.ofrs_grid_type in ['v']:
            grid_channels = 1
        elif self.ofrs_grid_type in ['geo','disctn']:
            grid_channels = 5
        else:
            raise KeyError

        self.of = nn.Sequential(
            _make_pad(1),
            nn.Conv2d(in_channels+grid_channels,mid_channels,kernel_size=3,bias=not norm),
            _make_norm(norm,mid_channels),
            _make_act(act),
            nn.Conv2d(mid_channels,2,1)
        )
        if in_channels == out_channels and not force:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels,out_channels,1)

    def register_embed(self,shape):
        b,_,h,w = shape
        if self.ofrs_grid_type == 'cos':
            uv = uv_grid(h,w).view(1,2,h,w)
            grid = torch.cat([torch.cos(uv[:,:1]),torch.cos(uv[:,1:]),torch.cos(uv[:,:1])*torch.cos(uv[:,1:])],dim=1)
            self.ofrs_grid = grid.expand([b,-1,-1,-1])
        elif self.ofrs_grid_type == 'sin':
            uv = uv_grid(h,w).view(1,2,h,w)
            grid = torch.cat([torch.sin(uv[:,:1]),torch.sin(uv[:,1:]),torch.sin(uv[:,:1])*torch.sin(uv[:,1:])],dim=1)
            self.ofrs_grid = grid.expand([b,-1,-1,-1])
        elif self.ofrs_grid_type == 'xyz':
            xyz = xyz_grid(h,w,dim=0).view(1,3,h,w)
            self.ofrs_grid = xyz.expand([b,-1,-1,-1])
        elif self.ofrs_grid_type == 'uv':
            uv = uv_grid(h,w).view(1,2,h,w)
            self.ofrs_grid = uv.expand([b,-1,-1,-1])
        elif self.ofrs_grid_type == 'v':
            uv = uv_grid(h,w).view(1,2,h,w)
            self.ofrs_grid = uv[1:].expand([b,-1,-1,-1])
        elif self.ofrs_grid_type == 'geo':
            uv = uv_grid(h,w).view(1,2,h,w)
            grid = torch.cat([torch.sin(uv[:,:1]),torch.cos(uv[:,:1]),torch.sin(uv[:,1:]),torch.cos(uv[:,1:]),torch.cos(uv[:,:1])*torch.cos(uv[:,1:])],dim=1)
            self.ofrs_grid = grid.expand([b,-1,-1,-1])
        elif self.ofrs_grid_type == 'disctn':
            uv = uv_grid(h,w).view(1,2,h,w)
            grid = torch.cat([torch.sin(uv[:,:1]),torch.cos(uv[:,:1]),torch.sin(uv[:,1:]),torch.cos(uv[:,1:]),torch.sin(uv[:,:1])*torch.sin(uv[:,1:])],dim=1)
            self.ofrs_grid = grid.expand([b,-1,-1,-1])
        else:
            raise KeyError

    def wraped_sample(self,x):

        b,_,h,w = x.shape

        if self.ofrs_grid is None or not self.ofrs_grid.shape[0] == b or not tuple(self.ofrs_grid.shape[-2:]) == tuple(x.shape[-2:]):
            self.register_embed(x.shape)

        if self.xy_grid is None or not tuple(self.xy_grid.shape[-2:]) == tuple(x.shape[-2:]):
            self.xy_grid = xy_grid(h,w).view(1,2,h,w) # indexing='xy'

        of_input = torch.cat([x,self.ofrs_grid],dim=1)
        of = self.of(of_input) # [b,2,h,w]
        of = of + self.xy_grid # [b,2,h,w]
        # print(x.shape,of.shape)

        of = of.permute([0,2,3,1]) #[b,h,w,2]
        y = spherical_grid_sample(x,of,inplace=False,indexing='xy')

        return y
    
    def forward(self,x):
        x = self.pad(x)
        x = F.interpolate(x,scale_factor=self.scale,mode='bilinear',align_corners=False)
        x = x[:,:,self.scale:-self.scale,self.scale:-self.scale]
        y = self.wraped_sample(x)
        y = self.conv(y)
        return y

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

def spherical_grid_sample(x,grid,inplace=False,indexing='ij'):
    B,C,H,W = x.shape
    assert len(grid.shape) == 4 and grid.shape[-1] == 2
    # assert tuple(grid.shape) == (B,H,W,2)

    grid_reminder = spherical_reminder(grid,[H,W],inplace=inplace,indexing=indexing)
    if indexing == 'ij':
        grid_h = grid_reminder[...,0]
        grid_w = grid_reminder[...,1]
    else:
        grid_h = grid_reminder[...,1]
        grid_w = grid_reminder[...,0]

    x = torch.cat([x,x[...,:1]],dim=-1)
    grid = torch.stack([grid_w/W*2-1,grid_h/(H-1)*2-1],dim=-1)
    y = F.grid_sample(x,grid,align_corners=True)
    return y
