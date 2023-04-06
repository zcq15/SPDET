import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from packaging.version import Version
import sys

sys.path.append('.')
sys.path.append('..')

from ..modules import _make_pad
from .ssim_thirdparty import ssim_loss
from ..geometry.imaging import CETransform

################################################################################

def cosine_similarity(x,y):
    return 1-F.cosine_similarity(x,y)

def similarity(x,y,kernel_size=7, std=1.5, mode='gaussian',pad=None,reduce=False):
    if pad is None: pad = _make_pad(kernel_size//2)
    sim = (1-ssim_loss(x, y, kernel_size=kernel_size, std=std, mode=mode, pad=pad))/2
    if reduce and not sim.shape[1] == 1: sim = sim.mean(dim=1,keepdim=True)
    return sim

def photometric(x,y,alpha=0.85,kernel_size=7, std=1.5, mode='gaussian',pad=None):
    assert alpha>=0 and alpha<=1.0
    sim = similarity(x, y, kernel_size=kernel_size, std=std, mode=mode, pad=pad, reduce=False)
    l1 = (x-y).abs()
    return sim*alpha+l1*(1-alpha)

class Photometric(torch.nn.Module):
    def __init__(self, branch='tar_rgb', target='tar_rgb',valid='src_valid', alpha=0.85,
        kernel_size=7, std=1.5, mode='gaussian', area=None):
        super().__init__()
        self.branch = branch
        self.target = target
        self.valid = valid
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.std = std
        self.mode = mode
        self.area = area
        self.pad_ssim = _make_pad(kernel_size//2)
        self.pad = _make_pad(1)

    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch]
        target = _input[self.target]

        b,_,h,w = target.shape

        with torch.no_grad():
            if self.valid in _input:
                mask_in = _input[self.valid]
            else:
                mask_in = torch.ones([b,1,h,w],dtype=torch.bool).cuda()
            if self.valid in _output:
                mask_out = _output[self.valid]
            else:
                mask_out = torch.ones([b,1,h,w],dtype=torch.bool).cuda()

            mask = (mask_in*mask_out).type(torch.float32)

        pred = pred*mask
        target = target*mask

        loss = photometric(pred, target, alpha=self.alpha, kernel_size=self.kernel_size, std=self.std, mode=self.mode, pad=self.pad_ssim)

        loss *= mask
        count = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
        loss = torch.mean(torch.sum(loss, dim=[1, 2, 3], keepdim=True) / count)
        if torch.isinf(loss) or torch.isnan(loss):
            return torch.tensor(0,dtype=torch.float32).cuda()
        else:
            return loss

class Perceptual(torch.nn.Module):
    def __init__(self, branch='tar_rgb', target='tar_rgb',valid='src_valid',
                vgg='vgg19', view='cube', dist='l1',
                indices=[2, 7, 12, 21, 30],
                weights=[1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
    ):
        super().__init__()
        self.branch = branch
        self.target = target
        self.valid = valid
        if Version(torch.__version__) >= Version('1.12.0'):
            self.vgg = getattr(torchvision.models,vgg)(weights=torchvision.models.VGG19_Weights.DEFAULT).features.cuda()
        else:
            self.vgg = getattr(torchvision.models,vgg)(pretrained=True).features.cuda()
        if dist == 'l1':
            self.metric = F.l1_loss
        elif dist == 'cos':
            self.metric = cosine_similarity
        else:
            raise KeyError

        self.indices = indices
        self.weights = {i:w for i,w in zip(indices,weights)}
        self.register_buffer('x_mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]).cuda())
        self.register_buffer('x_std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]).cuda())
        self.view = view
        self.cetrans = None
        self.imgsize = None

        if self.view == 'equi':
            self.pad = _make_pad(1)
            for layer in self.vgg:
                if isinstance(layer,torch.nn.Conv2d):
                    layer.padding=(0,0)

        for par in self.vgg.parameters():
            par.requires_grad = False

    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0,dtype=torch.float32).cuda()

        pred = _output[self.branch]
        target = _input[self.target]

        b,_,h,w = target.shape

        if self.view == 'cube' and not self.imgsize == (h,w):
            self.imgsize = (h,w)
            self.cetrans = CETransform(h,w,h//2).cuda()

        with torch.no_grad():
            if self.valid in _input:
                mask_in = _input[self.valid]
            else:
                mask_in = torch.ones([b,1,h,w],dtype=torch.bool).cuda()
            if self.valid in _output:
                mask_out = _output[self.valid]
            else:
                mask_out = torch.ones([b,1,h,w],dtype=torch.bool).cuda()

            mask = (mask_in*mask_out).type(torch.float32)

        pred = pred*mask
        target = target*mask

        if self.view == 'cube':
            pred = self.cetrans(pred,mode='e2c')
            target = self.cetrans(target,mode='e2c')

        loss = 0.0
        weight = 0.0

        feat_y = (pred - self.x_mean) / self.x_std
        feat_t = (target - self.x_mean) / self.x_std

        indices_max = np.max(self.indices)
        for l in range(len(self.vgg)):
            if self.view == 'equi' and isinstance(self.vgg[l],torch.nn.Conv2d):
                feat_y = self.pad(feat_y)
                feat_t = self.pad(feat_t)
            feat_y = self.vgg[l](feat_y)
            feat_t = self.vgg[l](feat_t)
            if l in self.indices:
                loss = loss + self.metric(feat_y.clone(),feat_t.clone()).mean()*self.weights[l]
                weight = weight + self.weights[l]
            if l == indices_max:
                break
        return loss/weight

class UVEdge(torch.nn.Module):
    def __init__(self, branch='tar_uv_scatter', target='source', grid='sobel',
        kernel_size=7, std=1.5, mode='gaussian'):
        super().__init__()
        self.branch = branch
        self.target = target
        self.kernel_size = kernel_size
        self.std = std
        self.mode = mode
        self.pad_ssim = _make_pad(kernel_size//2)
        self.pad = _make_pad(1)

        self.sobel = torch.tensor([[[[0,0,0],[-1,0,1],[0,0,0]]],[[[0,-1,0],[0,0,0],[0,1,0]]]]).type(torch.float32).cuda()
        self.laplacian = torch.tensor([[[[0,0,0],[1,-2,1],[0,0,0]]],[[[0,1,0],[0,-2,0],[0,1,0]]]]).type(torch.float32).cuda()
        assert grid in ['laplacian','sobel']
        self.grid = self.laplacian if grid == 'laplacian' else self.sobel

    def rgb2gray(self,rgb):
        return (0.299*rgb[:,0] + 0.587*rgb[:,1] + 0.114*rgb[:,2]).unsqueeze(1)

    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch]
        target = _input[self.target]

        assert pred.shape[1] == 2
        u,v = pred[:,:1], pred[:,1:]
        pred = torch.cos(u).abs()*torch.cos(v).abs()

        b,_,h,w = pred.shape
        pred = self.pad(pred.view(b,1,h,w))
        grad = torch.nn.functional.conv2d(pred,self.grid)
        grad = grad.view(b,2,h,w).abs()
        pred = grad.norm(p=2,dim=1,keepdim=True)

        target = self.rgb2gray(target)

        b,_,h,w = target.shape
        target = self.pad(target.view(b,1,h,w))
        grad = torch.nn.functional.conv2d(target,self.grid)
        grad = grad.view(b,2,h,w).abs()
        grad_u = grad[:,:1]*torch.sin(u)*torch.cos(v)
        grad_v = grad[:,1:]*torch.cos(u)*torch.sin(v)
        grad = torch.cat([grad_u,grad_v],dim=1)
        target = grad.norm(p=2,dim=1,keepdim=True)

        loss = similarity(pred, target, kernel_size=self.kernel_size, std=self.std, mode=self.mode, pad=self.pad_ssim)

        return loss.mean()

class UVSmooth(torch.nn.Module):
    def __init__(self, branch='tar_uv_scatter', target='source', grid='sobel'):
        super().__init__()
        self.branch = branch
        self.target = target
        self.pad = _make_pad(1)

        self.sobel = torch.tensor([[[[0,0,0],[-1,0,1],[0,0,0]]],[[[0,-1,0],[0,0,0],[0,1,0]]]]).type(torch.float32).cuda()
        self.laplacian = torch.tensor([[[[0,0,0],[1,-2,1],[0,0,0]]],[[[0,1,0],[0,-2,0],[0,1,0]]]]).type(torch.float32).cuda()
        assert grid in ['laplacian','sobel']
        self.grid = self.laplacian if grid == 'laplacian' else self.sobel

    def rgb2gray(self,rgb):
        return (0.299*rgb[:,0] + 0.587*rgb[:,1] + 0.114*rgb[:,2]).unsqueeze(1)

    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch]
        target = _input[self.target]

        assert pred.shape[1] == 2
        u,v = pred[:,:1], pred[:,1:]
        pred = torch.cos(u).abs()*torch.cos(v).abs()

        b,_,h,w = pred.shape
        pred = self.pad(pred.view(b,1,h,w))
        grad = torch.nn.functional.conv2d(pred,self.grid)
        grad = grad.view(b,2,h,w).abs()
        pred = grad.norm(p=2,dim=1,keepdim=True)

        target = self.rgb2gray(target)

        b,_,h,w = target.shape
        target = self.pad(target.view(b,1,h,w))
        grad = torch.nn.functional.conv2d(target,self.grid)
        grad = grad.view(b,2,h,w).abs()
        grad_u = grad[:,:1]*torch.sin(u)*torch.cos(v)
        grad_v = grad[:,1:]*torch.cos(u)*torch.sin(v)
        grad = torch.cat([grad_u,grad_v],dim=1)
        target = grad.norm(p=2,dim=1,keepdim=True)

        smooth = pred*torch.exp(-target)
        return smooth.mean()*h

class Smooth(torch.nn.Module):
    def __init__(self, branch='src_depth', target='source', grid='sobel'):
        super().__init__()
        self.branch = branch
        self.target = target
        self.pad = _make_pad(1)

        self.sobel = torch.tensor([[[[0,0,0],[-1,0,1],[0,0,0]]],[[[0,-1,0],[0,0,0],[0,1,0]]]]).type(torch.float32).cuda()
        self.laplacian = torch.tensor([[[[0,0,0],[1,-2,1],[0,0,0]]],[[[0,1,0],[0,-2,0],[0,1,0]]]]).type(torch.float32).cuda()
        assert grid in ['laplacian','sobel']
        self.grid = self.laplacian if grid == 'laplacian' else self.sobel

    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch]
        target = _input[self.target]

        b,c,h,w = pred.shape
        pred = pred.view(b*c,1,h,w)
        grad = torch.nn.functional.conv2d(pred,self.grid)
        pred = grad.view(b,c*2,h-2,w-2).norm(p=2,dim=1,keepdim=True)
        
        b,c,h,w = target.shape
        target = target.view(b*c,1,h,w)
        grad = torch.nn.functional.conv2d(target,self.grid)
        target = grad.view(b,c*2,h-2,w-2).norm(p=2,dim=1,keepdim=True)

        smooth = pred*torch.exp(-target)
        return smooth.mean()

class BerHu(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',valid='src_valid',th=0.2,min=0,max=10.0):
        super(BerHu,self).__init__()
        self.branch = branch
        self.target = target
        self.valid = valid
        self.th = th
        self.min = min
        self.max = max

    def forward(self,_output,_input):
        if not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if self.valid in _input:
            mask = _input[self.valid]
        else:
            mask = torch.ones_like(target,dtype=torch.bool,device=target.device)

        delta = (pred-target).abs()
        b,_,_,_ = _input['src_rgb'].shape
        if not mask is None:
            delta[torch.logical_not(mask)] = 0

        with torch.no_grad():
            T = delta.detach().max(dim=2,keepdim=True).values.max(dim=3,keepdim=True).values*self.th # B,1,1,1
            mask1 = (delta<=T)
            mask2 = (delta>T)
            if not mask is None:
                mask1 = mask1*mask
                mask2 = mask2*mask
        loss1 = delta[mask1]
        loss2 = ((delta.pow(2)+T.pow(2))/(2*T))[mask2]
        loss = (loss1.sum()+loss2.sum())/(loss1.numel()+loss2.numel())
        return loss

class L1(torch.nn.Module):
    def __init__(self, branch='rgb_rays', target='rgb_rays',valid='src_valid'):
        super().__init__()
        self.branch = branch
        self.target = target
        self.valid = valid

    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0,dtype=torch.float32).cuda()
    
        pred = _output[self.branch]
        target = _input[self.target]

        if self.valid in _input:
            mask = _input[self.valid]
            if not tuple(mask.shape) == tuple(target.shape):
                mask = mask.expand_as(target)
        else:
            mask = torch.ones_like(target,dtype=torch.bool,device=target.device)

        loss = torch.tensor(0,dtype=torch.float32).cuda()
        for b in range(_input[self.target].shape[0]):
            m = mask[b]
            if torch.any(m):
                p = pred[b][m]
                t = target[b][m]
                loss += (p-t).abs().mean()
            else:
                loss += 0*pred[b].abs().mean()
        if _input[self.target].shape[0] > 0:
            loss = loss/_input[self.target].shape[0]

        return loss

class PSNR(torch.nn.Module):
    def __init__(self, branch='tar_rgb', target='tar_rgb', eps=1e-5):
        super().__init__()
        self.branch = branch
        self.target = target
        self.eps = eps

    def forward(self,_output,_input):
        with torch.no_grad():
            if not self.branch in _output or not self.target in _input:
                return torch.tensor(0,dtype=torch.float32).cuda()
            pred = _output[self.branch].clone().detach()
            target = _input[self.target].clone().detach()

            b,c,h,w = pred.shape

            mse = (pred-target).pow(2)

            psnr = 0
            for i in range(b):
                psnr += 10*torch.log10((1.0**2+self.eps**2)/(mse[i].mean()+self.eps**2))

            return psnr/b
