'''
    PyTorch implementation of https://github.com/google/layered-scene-inference
    accompanying the paper "Layer-structured 3D Scene Inference via View Synthesis", 
    ECCV 2018 https://shubhtuls.github.io/lsi/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

import sys
sys.path.append('.')
sys.path.append('..')

from ..gargs import _args
from .geometry import sp2car,car2sp,uv2xy,xyz_grid
from ..modules import spherical_reminder,spherical_upsample,spherical_grid_sample
from pytorch3d import transforms as T

def cvt_pose2proj(pose):
    if tuple(pose.shape[1:]) == (6,1,1):
        pose = pose[:,:,0,0]
    elif tuple(pose.shape[1:]) == (6,):
        pass
    else:
        raise ValueError
    translate = pose[:,0:3]
    rotate = T.axis_angle_to_matrix(pose[:,3:6])
    b,_ = pose.shape
    proj = torch.zeros([b,3,4],dtype=torch.float32,device=pose.device)
    proj[:,0:3,0:3] = rotate
    proj[:,0:3,3] = -translate
    return proj

def __splat__(values, coords, splatted):
    b, c, h, w = splatted.size()
    uvs = coords
    u = uvs[:, 0, :, :].unsqueeze(1)
    v = uvs[:, 1, :, :].unsqueeze(1)
    
    u0 = torch.floor(u)
    u1 = u0 + 1
    v0 = torch.floor(v)
    v1 = v0 + 1

    u0_safe = torch.clamp(u0, 0.0, w-1)
    v0_safe = torch.clamp(v0, 0.0, h-1)
    u1_safe = torch.clamp(u1, 0.0, w-1)
    v1_safe = torch.clamp(v1, 0.0, h-1)

    u0_w = (u1 - u) * (u0 == u0_safe).detach().type(values.dtype)
    u1_w = (u - u0) * (u1 == u1_safe).detach().type(values.dtype)
    v0_w = (v1 - v) * (v0 == v0_safe).detach().type(values.dtype)
    v1_w = (v - v0) * (v1 == v1_safe).detach().type(values.dtype)

    top_left_w = u0_w * v0_w
    top_right_w = u1_w * v0_w
    bottom_left_w = u0_w * v1_w
    bottom_right_w = u1_w * v1_w

    weight_threshold = 1e-3
    top_left_w *= (top_left_w >= weight_threshold).detach().type(values.dtype)
    top_right_w *= (top_right_w >= weight_threshold).detach().type(values.dtype)
    bottom_left_w *= (bottom_left_w >= weight_threshold).detach().type(values.dtype)
    bottom_right_w *= (bottom_right_w >= weight_threshold).detach().type(values.dtype)

    for channel in range(c):
        top_left_values = values[:, channel, :, :].unsqueeze(1) * top_left_w
        top_right_values = values[:, channel, :, :].unsqueeze(1) * top_right_w
        bottom_left_values = values[:, channel, :, :].unsqueeze(1) * bottom_left_w
        bottom_right_values = values[:, channel, :, :].unsqueeze(1) * bottom_right_w

        top_left_values = top_left_values.reshape(b, -1)
        top_right_values = top_right_values.reshape(b, -1)
        bottom_left_values = bottom_left_values.reshape(b, -1)
        bottom_right_values = bottom_right_values.reshape(b, -1)

        top_left_indices = (u0_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        top_right_indices = (u1_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        bottom_left_indices = (u0_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        bottom_right_indices = (u1_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        
        splatted_channel = splatted[:, channel, :, :].unsqueeze(1)
        splatted_channel = splatted_channel.reshape(b, -1)
        splatted_channel.scatter_add_(1, top_left_indices, top_left_values)
        splatted_channel.scatter_add_(1, top_right_indices, top_right_values)
        splatted_channel.scatter_add_(1, bottom_left_indices, bottom_left_values)
        splatted_channel.scatter_add_(1, bottom_right_indices, bottom_right_values)
    splatted = splatted.reshape(b, c, h, w)

def __weighted_average_splat__(feature, weights, epsilon=1e-8):
    zero_weights = (weights <= epsilon).detach().type(feature.dtype)
    return feature / (weights + epsilon * zero_weights)

def __depth_distance_weights__(depth, max_depth=10.0):
    with torch.no_grad():
        weights = 1.0 / torch.exp(depth / max_depth)
    return weights

def rotate_angle(x,offset):
    offset = offset.view([x.shape[0]]+[1]*(len(x.shape)-1))
    offset = offset/(torch.pi*2)*x.shape[-1]
    offset_floor = torch.floor(offset)
    offset_ceil = offset_floor+1
    w0 = offset_ceil-offset
    w1 = offset-offset_floor

    x0 = []
    x1 = []
    for b in range(x.shape[0]):
        x0.append(torch.roll(x[b],offset_floor[b].type(torch.int64).cpu().item(),dims=[-1]))
        x1.append(torch.roll(x[b],offset_ceil[b].type(torch.int64).cpu().item(),dims=[-1]))
    x0 = torch.stack(x0,dim=0)
    x1 = torch.stack(x1,dim=0)

    return x0*w0+x1*w1

def rotate_pixel(x,offset):
    offset = offset.view([x.shape[0]]+[1]*(len(x.shape)-1))
    offset_floor = torch.floor(offset)
    offset_ceil = offset_floor+1
    w0 = offset_ceil-offset
    w1 = offset-offset_floor

    x0 = []
    x1 = []
    for b in range(x.shape[0]):
        x0.append(torch.roll(x[b],offset_floor[b].type(torch.int64).cpu().item(),dims=[-1]))
        x1.append(torch.roll(x[b],offset_ceil[b].type(torch.int64).cpu().item(),dims=[-1]))
    x0 = torch.stack(x0,dim=0)
    x1 = torch.stack(x1,dim=0)

    return x0*w0+x1*w1

def scatter(feat,depth,pos,mask=None,upscale=int(1),downscale=int(1),th=10.0,rot=None,rot_first=False):
    assert len(feat.shape) == 4
    assert len(depth.shape) == 4
    assert len(pos.shape) in [2,3,4]
    if len(pos.shape) in [2,4]:
        if pos.shape[1] == 6:
            pos = cvt_pose2proj(pos)
        else:
            pos = pos.view(pos.shape[0],3,1,1)
    else:
        assert tuple(pos.shape[1:]) == (3,4)
    assert mask is None or len(mask.shape) == 4

    assert isinstance(upscale,int) and upscale > 0
    assert isinstance(downscale,int) and downscale > 0
    assert tuple(feat.shape[-2:]) == tuple(depth.shape[-2:])
    assert (feat.shape[-2]*upscale)%downscale == 0 and (feat.shape[-1]*upscale)%downscale == 0

    if torch.is_tensor(rot) and rot_first:
        feat = rotate_angle(feat,rot)
        depth = rotate_angle(depth,rot)
        if not mask is None:
            mask = rotate_angle(mask.type(torch.float32),rot) > 0.5

    xyz = sp2car(depth)
    if len(pos.shape) == 4:
        xyz = xyz - pos
    else:
        b,c,h,w = xyz.shape
        ones = torch.ones([b,1,h,w],dtype=torch.float32,device=xyz.device)
        xyz = torch.einsum('bmn,bnhw->bmhw',pos,torch.cat([xyz,ones],dim=1))
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    uv = car2sp(x,y,z,dim=1)[:,1:]

    if upscale > 1:
        feat = spherical_upsample(feat,upscale)
        depth = spherical_upsample(depth,upscale)

    b,c,h,w = feat.shape
    hd,wd = h//downscale,w//downscale
    if mask is None:
        mask = torch.ones([b,1,h,w],dtype=torch.float32).cuda()
    else:
        mask = mask.type(torch.float32)
        if upscale > 1: mask = spherical_upsample(mask,upscale)

    xyz = sp2car(depth)
    if len(pos.shape) == 4:
        xyz = xyz - pos
    else:
        b,c,h,w = xyz.shape
        ones = torch.ones([b,1,h,w],dtype=torch.float32,device=xyz.device)
        xyz = torch.einsum('bmn,bnhw->bmhw',pos,torch.cat([xyz,ones],dim=1))
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    ruv = car2sp(x,y,z)
    r,u,v = ruv[0],ruv[1],ruv[2]

    if torch.is_tensor(rot) and not rot_first:
        u = u + rot.view(b,1,1)

    coords = uv2xy(u,v,hd,wd,dim=-1) #[b,h,w,2]
    coords = spherical_reminder(coords,[hd,wd],indexing='xy').permute(0,3,1,2)

    splatted_feat = torch.zeros([b,c,hd,wd],dtype=torch.float32).cuda()
    splatted_wgts = torch.zeros([b,1,hd,wd],dtype=torch.float32).cuda()  
    weights = __depth_distance_weights__(r.unsqueeze(1), max_depth=th)
    __splat__(feat * weights * mask, coords, splatted_feat)
    __splat__(weights * mask, coords, splatted_wgts)
    recon = __weighted_average_splat__(splatted_feat, splatted_wgts)
    mask = (splatted_wgts > 1e-3).detach()
    
    return recon, mask, uv.contiguous(), coords.contiguous(), splatted_wgts


def scatter_depth(depth,pos,mask=None,upscale=int(1),downscale=int(1),th=10.0):
    xyz = sp2car(depth)
    xyz = xyz - pos
    r = torch.norm(xyz,p=2,dim=1,keepdim=True)
    depth,mask,_,_,_ = scatter(r,depth,pos,mask,upscale,downscale,th)
    return depth,mask


def inter(feat,depth,pos,mask=None,rot=None,rot_first=False):
    assert len(feat.shape) == 4
    assert len(depth.shape) == 4
    assert len(pos.shape) in [2,3,4]
    if len(pos.shape) in [2,4]:
        if pos.shape[1] == 6:
            pos = cvt_pose2proj(pos)
        else:
            pos = pos.view(pos.shape[0],3,1,1)
    else:
        assert tuple(pos.shape[1:]) == (3,4)
    assert mask is None or len(mask.shape) == 4
    assert tuple(feat.shape[-2:]) == tuple(depth.shape[-2:])

    b,c,h,w = feat.shape
    if mask is None:
        mask = torch.ones([b,1,h,w],dtype=torch.float32).cuda()
    else:
        mask = mask.type(torch.float32)

    if torch.is_tensor(rot) and rot_first:
        depth = rotate_angle(depth,rot)

    xyz = sp2car(depth)
    if len(pos.shape) == 4:
        xyz = xyz - pos
    else:
        b,c,h,w = xyz.shape
        ones = torch.ones([b,1,h,w],dtype=torch.float32,device=xyz.device)
        xyz = torch.einsum('bmn,bnhw->bmhw',pos,torch.cat([xyz,ones],dim=1))
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    ruv = car2sp(x,y,z)
    r,u,v = ruv[0],ruv[1],ruv[2]

    if torch.is_tensor(rot) and not rot_first:
        u = u + rot.view(b,1,1)

    coords = uv2xy(u,v,h,w,dim=-1) #[b,h,w,2]
    coords = spherical_reminder(coords,[h,w],indexing='xy')

    feature = spherical_grid_sample(feat,coords,indexing='xy')

    zeros = torch.zeros_like(feature)
    feature = torch.where(mask.expand(-1,c,-1,-1)>0,feature,zeros)

    return feature


def warp_at_novelview(src_feat,tar_hypos,tar_pos,order='cdhw'): # src -> tar
    assert len(src_feat.shape) == 4
    assert len(tar_hypos.shape) == 4
    assert len(tar_pos.shape) == 4
    # assert tuple(feat.shape[-2:]) == tuple(hypos.shape[-2:])

    B,N,H,W = tar_hypos.shape
    tar_hypos = tar_hypos.reshape(B*N,1,H,W)

    _,c,h,w = src_feat.shape
    src_feat = src_feat.unsqueeze(1).expand([-1,N,-1,-1,-1]).reshape(B*N,c,h,w)

    tar_pos = tar_pos.unsqueeze(1).expand([-1,N,-1,-1,-1]).reshape(B*N,3,1,1)

    xyz = sp2car(tar_hypos)
    xyz = xyz + tar_pos
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    ruv = car2sp(x,y,z)
    _,u,v = ruv[0],ruv[1],ruv[2]
    coords = uv2xy(u,v,h,w,dim=-1) #[bn,h,w,2]

    feature = spherical_grid_sample(src_feat,coords,indexing='xy')

    feature = feature.view(B,N,c,H,W)
    if order == 'cdhw':
        feature = feature.permute(0,2,1,3,4)

    return feature

def warp_here(tar_feat,src_hypos,tar_pos,order='cdhw'): # tar -> src
    assert len(tar_feat.shape) == 4
    assert len(src_hypos.shape) == 4
    assert len(tar_pos.shape) == 4

    B,N,H,W = src_hypos.shape
    src_hypos = src_hypos.reshape(B*N,1,H,W)

    _,c,h,w = tar_feat.shape
    tar_feat = tar_feat.unsqueeze(1).expand([-1,N,-1,-1,-1]).reshape(B*N,c,h,w)

    tar_pos = tar_pos.unsqueeze(1).expand([-1,N,-1,-1,-1]).reshape(B*N,3,1,1)

    xyz = sp2car(src_hypos)
    xyz = xyz - tar_pos
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    ruv = car2sp(x,y,z)
    _,u,v = ruv[0],ruv[1],ruv[2]
    coords = uv2xy(u,v,h,w,dim=-1) #[bn,h,w,2]

    feature = spherical_grid_sample(tar_feat,coords,indexing='xy')

    feature = feature.view(B,N,c,H,W)
    if order == 'cdhw':
        feature = feature.permute(0,2,1,3,4)

    return feature

def norm_at_novelview(feat,hypos,pos):
    assert len(feat.shape) == 4
    assert len(hypos.shape) == 4
    assert len(pos.shape) == 4
    # assert tuple(feat.shape[-2:]) == tuple(hypos.shape[-2:])

    B,N,H,W = hypos.shape
    hypos = hypos.view(B*N,1,H,W)

    _,c,h,w = feat.shape

    pos = pos.unsqueeze(1).expand([-1,N,-1,-1,-1]).reshape(B*N,3,1,1)

    norm_tar = xyz_grid(H,W,dim=0).unsqueeze(0).expand(B*N,-1,-1,-1)
    xyz = norm_tar*hypos
    xyz = xyz + pos
    norm_src = xyz/torch.norm(xyz,dim=1,keepdim=True)
    return norm_src.view(B,N,3,H,W),norm_tar.view(B,N,3,H,W)


def init_hypos(shape,num=64,start=0,end=10.0):
    with torch.no_grad():
        b,_,h,w = shape
        hypos = torch.linspace(start, end, num, dtype=torch.float32).cuda()
        hypos = hypos.view(1,num,1,1).expand([b,-1,h,w])
        return hypos


def raw2outputs(raw, hypos, rays=None, raw_noise_std=0):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        hypos: [num_rays, num_samples along ray]. Integration time.
        rays: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = hypos[...,1:] - hypos[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    if not rays is None:
        dists = dists * torch.norm(rays[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape).cuda() * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * hypos, -1)

    return rgb_map, depth_map, weights

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples).cuda()
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).cuda()

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u).cuda()

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples