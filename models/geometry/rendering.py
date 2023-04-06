'''
    PyTorch implementation of https://github.com/google/layered-scene-inference
    accompanying the paper "Layer-structured 3D Scene Inference via View Synthesis", 
    ECCV 2018 https://shubhtuls.github.io/lsi/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from plyfile import PlyData,PlyElement
from packaging.version import Version

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

def map_eye_to_order(eye,dtype='single',device=None):
    if torch.is_tensor(eye):
        order = eye.type(torch.int64)
    else:
        if isinstance(eye,str):
            eye = [eye]
        eye_to_order = {'left':int(-1), 'right':int(1), 'center':int(0)}
        order = [eye_to_order[_] for _ in eye]
        order = torch.tensor(order)
    if dtype == 'single':
        order = order.view(1,1)
    if dtype == 'batch':
        order = order.view(-1,1,1)
    if dtype == 'multi':
        order = order.view(-1,1,1,1)
    if not device is None:
        order = order.to(device)
    return order

def map_basel_to_tensor(basel,dtype='single',device=None):
    if not torch.is_tensor(basel):
        basel = torch.tensor(basel).type(torch.float32)
    assert torch.all(basel>=0)
    if dtype == 'single':
        basel = basel.view(1,1)
    if dtype == 'batch':
        basel = basel.view(-1,1,1)
    if dtype == 'multi':
        basel = basel.view(-1,1,1,1)
    if not device is None:
        basel = basel.to(device)
    return basel

def uniform_depth_to_xyz(depth,basel=0.,eye='center',dtype='single'):

    assert dtype in ['single', 'batch', 'multi']
    if dtype == 'single': # [H,W]
        assert len(depth.size()) == 2
    if dtype == 'batch': # [B,H,W]
        assert len(depth.size()) in [3,4]
        if len(depth.size()) == 4 and depth.size()[-3] == 1:
            depth = depth.squeeze(-3)
    if dtype == 'multi': # [B,D,H,W]
        assert len(depth.size()) == 4

    order = map_eye_to_order(eye,dtype=dtype,device=depth.device)
    basel = map_basel_to_tensor(basel,dtype=dtype,device=depth.device)

    if torch.all(order==0):

        # print('erp sphere to xyz')

        h,w = depth.size()[-2:]
        if Version(torch.__version__) >= Version('1.10.0'):
            y,x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        else:
            y,x = torch.meshgrid(torch.arange(h), torch.arange(w))
        u = (x-(w-1)/2)/w*(2*np.pi)
        v = -(y-(h-1)/2)/h*np.pi
        x = -torch.cos(v)*torch.cos(u)
        y = torch.cos(v)*torch.sin(u)
        z = torch.sin(v)
        xyz = torch.stack([x,y,z],dim=0)
        xyz = torch.einsum('ijk,...jk->...ijk',xyz.to(depth.device),depth)
        return xyz
    
    else:

        # print('ods sphere to xyz')

        assert not torch.any(order==0)
        
        h,w = depth.size()[-2:]
        assert order in [-1,1]
        if Version(torch.__version__) >= Version('1.10.0'):
            y,x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        else:
            y,x = torch.meshgrid(torch.arange(h), torch.arange(w))
        u = (x.to(depth.device)-(w-1)/2)/w*(2*np.pi)
        v = -(y.to(depth.device)-(h-1)/2)/h*np.pi

        # ray direction
        rx = torch.cos(v)*torch.sin(u)
        ry = torch.cos(v)*torch.cos(u)
        rz = torch.sin(v)

        # ray center
        cx = -torch.cos(u)*basel*order
        cy = torch.sin(u)*basel*order
        cz = torch.zeros_like(v)

        a = rx*rx + ry*ry + rz*rz
        b = 2*(rx*cx + ry*cy + rz*cz)
        c = cx*cx + cy*cy + cz*cz - depth*depth

        disc = torch.square(b) - 4 * a * c
        t = (-b + torch.sqrt(disc)) / (2 * a)

        # rotate 90-degree
        x = -(cy + t*ry)
        y = cx + t*rx
        z = cz + t*rz
        xyz = torch.stack([x,y,z],dim=-3)

        return xyz

def uniform_xyz_to_sphere(points,h,w,basel=0.,eye='center',dtype='single',ret_depth=False):
    x_spsc,y_spsc,z_spsc = [_.squeeze(-3) for _ in points.split(1,dim=-3)]

    order = map_eye_to_order(eye,dtype=dtype,device=points.device)
    basel = map_basel_to_tensor(basel,dtype=dtype,device=points.device)

    if torch.all(order==0):

        # print('xyz to erp sphere')

        x = x_spsc
        y = y_spsc
        z = z_spsc

        theta = torch.atan2(y,-x)
        phi = torch.atan2(z,torch.norm(torch.stack([x,y],dim=-1),p=2,dim=-1))

        if ret_depth:
            depth = torch.norm(torch.stack([x,y,z],dim=-1),p=2,dim=-1)

        coord_x = theta/(2*np.pi)*w+(w-1)/2
        coord_y = -phi/np.pi*h+(h-1)/2

    # else: # migrate from ReplicaSDK

    #     # print('xyz to ods sphere')
    #     assert not torch.any(order==0)

    #     x = y_spsc
    #     y = z_spsc
    #     z = -x_spsc

    #     xx = torch.square(x)
    #     zz = torch.square(z)
    #     r = torch.sqrt(xx + zz) - 0.002
    #     r = torch.where(r>0,r,torch.zeros_like(r))

    #     # assert torch.all(basel<r)
    #     r = torch.where(basel<=r,basel,r)

    #     f = torch.square(r) - xx - zz

    #     z_larger_x = z.abs() > x.abs()
    #     px = torch.where(z_larger_x,x,z)
    #     pz = torch.where(z_larger_x,z,x)

    #     pxx = torch.square(px)
    #     pzz = torch.square(pz)
    #     a = 1 + pxx / pzz
    #     b = -2 * f * px / pzz
    #     c = f + f * f / pzz
    #     # disc = torch.square(b) - 4 * a * c
    #     disc = -4 * torch.square(r) * f / pzz

    #     # print((disc>=0).type(torch.float32).mean())

    #     assert torch.all(disc >= 0)

    #     theta_erp = torch.atan2(x, z)
    #     phi_erp = torch.atan2(y, torch.sqrt(xx+zz))

    #     if ret_depth:
    #         depth_erp = torch.norm(torch.stack([x,y,z],dim=-1),p=2,dim=-1)

    #     s = torch.sign(pz) * torch.sqrt(disc)
    #     s = torch.where(z_larger_x, s, -s)
    #     s = - order * s
    #     dx = (-b + s) / (2 * a)
    #     dz = (f - px * dx) / pz
        
    #     dx_final = torch.where(z_larger_x,-dx,-dz)
    #     dy_final = y
    #     dz_final = torch.where(z_larger_x,-dz,-dx)

    #     theta_ods = torch.atan2(dx_final, dz_final)
    #     phi_ods = torch.atan2(dy_final, torch.sqrt(torch.square(dx_final)+torch.square(dz_final)))

    #     if ret_depth:
    #         depth_ods = torch.norm(torch.stack([dx_final,dy_final,dz_final],dim=-1),p=2,dim=-1)

    #     select_ods = disc >= 0

    #     theta = torch.where(select_ods, theta_ods, theta_erp)
    #     phi = torch.where(select_ods, phi_ods, phi_erp)

    #     if ret_depth:
    #         depth = torch.where(select_ods, depth_ods, depth_erp)

    #     coord_x = theta/(2*np.pi)*w+(w-1)/2
    #     coord_y = -phi/np.pi*h+(h-1)/2

    else: # migrate from MatryODShka paper

        # print('xyz to ods sphere')
        assert not torch.any(order==0)

        depth = None

        x = y_spsc
        y = z_spsc
        z = -x_spsc

        r = torch.sqrt(torch.square(x)+torch.square(z))
        r = torch.where(r>basel, r, basel.expand_as(r))

        theta = order * torch.arcsin(basel / r ) + torch.atan2(x, z)
        phi = torch.atan2(y, torch.sqrt(torch.square(r)-torch.square(basel)) )

        coord_x = theta/(2*np.pi)*w+(w-1)/2
        coord_y = -phi/np.pi*h+(h-1)/2

    # else: # migrate from MatryODShka code

    #     # print('xyz to ods sphere')
    #     assert not torch.any(order==0)

    #     depth = None

    #     x = -x_spsc
    #     y = z_spsc
    #     z = -y_spsc

    #     r = basel

    #     f = r * r - (torch.square(x) + torch.square(z))
    #     z_larger_x = torch.greater(torch.abs(z), torch.abs(x))
    #     px = torch.where(z_larger_x, x, z)
    #     pz = torch.where(z_larger_x, z, x)

    #     # Solve quadratic
    #     pz_square = torch.square(pz)
    #     a = 1 + torch.square(px) / pz_square
    #     b = -2 * f * px / pz_square
    #     c = f + torch.square(f) / pz_square
    #     disc = torch.square(b) - 4 * a * c

    #     # print(['pzz:', pz_square.abs().min()])

    #     # Direction vector from point
    #     s = -order * torch.sign(pz) * torch.sqrt(disc)
    #     s = torch.where(z_larger_x, s, -s)

    #     dx = (-b + s) / (2 * a )
    #     dz = (f - px * dx) / (pz)

    #     # print(['a:', a.abs().min()])
    #     # print(['pz:', pz.abs().min()])

    #     dx_final = torch.where(z_larger_x, -dx, -dz)
    #     dz_final = torch.where(z_larger_x, -dz, -dx)
    #     dx = dx_final
    #     dz = dz_final
    #     dy = y

    #     # Angles from direction vector
    #     theta = -torch.atan2(dz, dx)
    #     phi = torch.atan2(dy, torch.sqrt(torch.square(dx) + torch.square(dz)))
    #     nan_mask = torch.isnan(phi) | torch.isinf(phi)
    #     phi = torch.where(nan_mask, torch.ones_like(phi), phi)

    #     pos_phi = torch.ones_like(dx) * np.pi/2
    #     neg_phi = torch.ones_like(dx) * np.pi/2 * -1.

    #     pos_phi_mask = torch.less_equal(phi, np.pi/2)
    #     neg_phi_mask = torch.greater_equal(phi, -np.pi/2)
    #     phi = torch.where(pos_phi_mask, phi, pos_phi)
    #     phi = torch.where(neg_phi_mask, phi, neg_phi)

    #     coord_x = theta/(2*np.pi)*w+(w-1)/2
    #     coord_y = -phi/np.pi*h+(h-1)/2

    xy_grid = torch.stack([coord_x,coord_y],dim=-1) # [b,h,w,2]
    xy_grid = spherical_reminder(xy_grid,[h,w],indexing='xy')

    if ret_depth:
        return xy_grid, depth
    else:
        return xy_grid

def uniform_interpolate(ref_img, tar_depth, mat=None, R=None, t=None, ref_basel=0., ref_eye='center', tar_basel=0., tar_eye='center', dtype='batch', order='dchw'):

    # reference image + target depth -> target image
    # camera matrix: target -> reference

    if ref_basel is None: ref_basel = 0.
    if ref_eye is None: ref_eye = 'center'
    if tar_basel is None: tar_basel =0.
    if tar_eye is None: tar_eye = 'center'
    if dtype is None: dtype = 'batch'

    if dtype == 'single':
        assert len(tar_depth.size()) == 2
        tar_depth = tar_depth.unsqueeze(0) # [B,H,W]
        ref_img = ref_img.unsqueeze(0) # [B,C,H,W]

        if not (mat is None):
            mat = mat.unsqueeze(1)
        if not (R is None):
            R = R.unsqueeze(1)
        if not (t is None):
            if t.size()[-1] == 3:
                t = t.unsqueeze(1)
            else:
                assert tuple(t.size()[-3:]) == (3,1,1)
                t = t.unsqueeze(1)

        if not torch.is_tensor(ref_basel):
            ref_basel = torch.tensor(ref_basel).type(torch.float32).view(1)
        if tuple(ref_basel.size()) == tuple():
            ref_basel = ref_basel.view(1)
        if isinstance(ref_eye,str):
            ref_eye = [ref_eye]
        if not torch.is_tensor(tar_basel):
            tar_basel = torch.tensor(tar_basel).type(torch.float32).view(1)
        if tuple(tar_basel.size()) == tuple():
            tar_basel = tar_basel.view(1)
        if isinstance(tar_eye,str):
            tar_eye = [tar_eye]

    if dtype == 'batch':
        assert len(tar_depth.size()) in [3,4]
        if len(tar_depth.size()) == 4 and tar_depth.size()[-3] == 1:
            tar_depth = tar_depth.squeeze(-3) # [B,H,W]
        B,H,W = tar_depth.size()

        if not torch.is_tensor(ref_basel):
            ref_basel = torch.tensor(ref_basel).type(torch.float32).view(1)
        if tuple(ref_basel.size()) == tuple():
            ref_basel = ref_basel.view(1)
        if isinstance(ref_eye,str):
            ref_eye = [ref_eye]*B
        if not torch.is_tensor(tar_basel):
            tar_basel = torch.tensor(tar_basel).type(torch.float32).view(1)
        if tuple(tar_basel.size()) == tuple():
            tar_basel = tar_basel.view(1)
        if isinstance(tar_eye,str):
            tar_eye = [tar_eye]*B

    if dtype == 'multi':
        assert len(tar_depth.size()) == 4
        B,D,H,W = tar_depth.size()
        _,c,h,w = ref_img.size()
        ref_img = ref_img.unsqueeze(1).expand(B,D,c,h,w).reshape(B*D,c,h,w)
        tar_depth = tar_depth.view(B*D,H,W)

        if not (mat is None):
            mat = mat.unsqueeze(1).expand(B,D,4,4).reshape(B*D,4,4)
        if not (R is None):
            R = R.unsqueeze(1).expand(B,D,4,4).reshape(B*D,4,4)
        if not (t is None):
            if t.size()[-1] == 3:
                t = t.unsqueeze(1).expand(B,D,3).reshape(B*D,3,1,1)
            else:
                assert tuple(t.size()[-3:]) == (3,1,1)
                t = t.unsqueeze(1).expand(B,D,3,1,1).reshape(B*D,3,1,1)

        if not torch.is_tensor(ref_basel):
            ref_basel = torch.tensor(ref_basel).type(torch.float32).view(1)
        if tuple(ref_basel.size()) == tuple():
            ref_basel = ref_basel.view(1)
        ref_basel = ref_basel.unsqueeze(1).expand(B,D).reshape(B*D)
        if isinstance(ref_eye,str):
            ref_eye = [ref_eye]*B
        ref_eye_multi = []
        for e in ref_eye:
            ref_eye_multi += [e]*D
        ref_eye = ref_eye_multi
        if not torch.is_tensor(tar_basel):
            tar_basel = torch.tensor(tar_basel).type(torch.float32).view(1)
        if tuple(tar_basel.size()) == tuple():
            tar_basel = tar_basel.view(1)
        tar_basel = tar_basel.unsqueeze(1).expand(B,D).reshape(B*D)
        if isinstance(tar_eye,str):
            tar_eye = [tar_eye]*B
        tar_eye_multi = []
        for e in tar_eye:
            tar_eye_multi += [e]*D
        tar_eye = tar_eye_multi

    xyz = uniform_depth_to_xyz(tar_depth,tar_basel,tar_eye,dtype='batch')
    
    if not (mat is None):
        ones = torch.ones_like(tar_depth)
        xyz = torch.cat([xyz,ones],dim=-3)
        assert tuple(mat.size()[-2:]) == (4,4)
        xyz = torch.einsum('...mn,...njk->...mjk',mat,xyz)[...,:3,:,:]
    else:
        if not (R is None):
            assert tuple(R.size()[-2:]) == (3,3)
            xyz = torch.einsum('...mn,...njk->...mjk',R,xyz)
        if not (t is None):
            xyz = xyz - t
    
    xy_grid = uniform_xyz_to_sphere(xyz,ref_img.size()[-2],ref_img.size()[-1],ref_basel,ref_eye,dtype='batch')

    tar_img = spherical_grid_sample(ref_img,xy_grid,clip=False,indexing='xy')

    if dtype == 'single':
        tar_img = tar_img.squeeze(0)
    elif dtype == 'multi':
        tar_img = tar_img.view(B,D,c,H,W)
        if order == 'cdhw':
            tar_img = tar_img.permute(0,2,1,3,4)

    return tar_img 

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

def uniform_scatter(ref_img, ref_depth, mat=None, R=None, t=None, ref_basel=0., ref_eye='center', tar_basel=0., tar_eye='center', dtype='batch', order='dchw', mask=None, upscale=2, downscale=2, th=10.0):

    # reference image + reference depth -> target image
    # camera matrix: reference -> target

    if dtype == 'single':
        assert len(ref_depth.size()) == 2
        ref_depth = ref_depth.unsqueeze(0) # [B,H,W]
        ref_img = ref_img.unsqueeze(0) # [B,C,H,W]

        if not (mat is None):
            mat = mat.unsqueeze(1)
        if not (R is None):
            R = R.unsqueeze(1)
        if not (t is None):
            if t.size()[-1] == 3:
                t = t.unsqueeze(1)
            else:
                assert tuple(t.size()[-3:]) == (3,1,1)
                t = t.unsqueeze(1)

        if not torch.is_tensor(ref_basel):
            ref_basel = torch.tensor(ref_basel).type(torch.float32).view(1)
        if tuple(ref_basel.size()) == tuple():
            ref_basel = ref_basel.view(1)
        if isinstance(ref_eye,str):
            ref_eye = [ref_eye]
        if not torch.is_tensor(tar_basel):
            tar_basel = torch.tensor(tar_basel).type(torch.float32).view(1)
        if tuple(tar_basel.size()) == tuple():
            tar_basel = tar_basel.view(1)
        if isinstance(tar_eye,str):
            tar_eye = [tar_eye]

    if dtype == 'batch':
        assert len(ref_depth.size()) in [3,4]
        if len(ref_depth.size()) == 4 and ref_depth.size()[-3] == 1:
            ref_depth = ref_depth.squeeze(-3) # [B,H,W]
        B,H,W = ref_depth.size()

        if not torch.is_tensor(ref_basel):
            ref_basel = torch.tensor(ref_basel).type(torch.float32).view(1)
        if tuple(ref_basel.size()) == tuple():
            ref_basel = ref_basel.view(1)
        if isinstance(ref_eye,str):
            ref_eye = [ref_eye]*B
        if not torch.is_tensor(tar_basel):
            tar_basel = torch.tensor(tar_basel).type(torch.float32).view(1)
        if tuple(tar_basel.size()) == tuple():
            tar_basel = tar_basel.view(1)
        if isinstance(tar_eye,str):
            tar_eye = [tar_eye]*B

    if dtype == 'multi':
        assert len(ref_depth.size()) == 4
        B,D,H,W = ref_depth.size()
        _,c,h,w = ref_img.size()
        ref_img = ref_img.unsqueeze(1).expand(B,D,c,h,w).reshape(B*D,c,h,w)
        ref_depth = ref_depth.view(B*D,H,W)

        if not (mat is None):
            mat = mat.unsqueeze(1).expand(B,D,4,4).reshape(B*D,4,4)
        if not (R is None):
            R = R.unsqueeze(1).expand(B,D,4,4).reshape(B*D,4,4)
        if not (t is None):
            if t.size()[-1] == 3:
                t = t.unsqueeze(1).expand(B,D,3).reshape(B*D,3,1,1)
            else:
                assert tuple(t.size()[-3:]) == (3,1,1)
                t = t.unsqueeze(1).expand(B,D,3,1,1).reshape(B*D,3,1,1)

        if not torch.is_tensor(ref_basel):
            ref_basel = torch.tensor(ref_basel).type(torch.float32).view(1)
        if tuple(ref_basel.size()) == tuple():
            ref_basel = ref_basel.view(1)
        ref_basel = ref_basel.unsqueeze(1).expand(B,D).reshape(B*D)
        if isinstance(ref_eye,str):
            ref_eye = [ref_eye]*B
        ref_eye_multi = []
        for e in ref_eye:
            ref_eye_multi += [e]*D
        ref_eye = ref_eye_multi
        if not torch.is_tensor(tar_basel):
            tar_basel = torch.tensor(tar_basel).type(torch.float32).view(1)
        if tuple(tar_basel.size()) == tuple():
            tar_basel = tar_basel.view(1)
        tar_basel = tar_basel.unsqueeze(1).expand(B,D).reshape(B*D)
        if isinstance(tar_eye,str):
            tar_eye = [tar_eye]*B
        tar_eye_multi = []
        for e in tar_eye:
            tar_eye_multi += [e]*D
        tar_eye = tar_eye_multi

    if upscale > 1:
        ref_img = spherical_upsample(ref_img,upscale)
        ref_depth = spherical_upsample(ref_depth.unsqueeze(1),upscale).squeeze(1)

    b,c,h,w = ref_img.shape
    
    if mask is None:
        mask = torch.ones([b,1,h,w],dtype=torch.float32).cuda()
    else:
        mask = mask.type(torch.float32)
        if upscale > 1: mask = spherical_upsample(mask,upscale)

    xyz = uniform_depth_to_xyz(ref_depth,tar_basel,tar_eye,dtype='batch')

    if not (mat is None):
        ones = torch.ones_like(ref_depth)
        xyz = torch.cat([xyz,ones],dim=-3)
        assert tuple(mat.size()[-2:]) == (4,4)
        xyz = torch.einsum('...mn,...njk->...mjk',mat,xyz)[...,:3,:,:]
    else:
        if not (R is None):
            assert tuple(R.size()[-2:]) == (3,3)
            xyz = torch.einsum('...mn,...njk->...mjk',R,xyz)
        if not (t is None):
            xyz = xyz - t
    
    b,c,h,w = ref_img.shape
    h,w = h//downscale,w//downscale

    xy_grid, tar_depth = uniform_xyz_to_sphere(xyz,h,w,ref_basel,ref_eye,dtype='batch', ret_depth=True)

    coords = xy_grid.permute(0,3,1,2) # [b,2,h,w]

    splatted_feat = torch.zeros([b,c,h,w],dtype=torch.float32).cuda()
    splatted_wgts = torch.zeros([b,1,h,w],dtype=torch.float32).cuda()  
    weights = __depth_distance_weights__(tar_depth.unsqueeze(1), max_depth=th)
    __splat__(ref_img * weights * mask, coords, splatted_feat)
    __splat__(weights * mask, coords, splatted_wgts)
    recon = __weighted_average_splat__(splatted_feat, splatted_wgts)
    mask = (splatted_wgts > 1e-3).detach()
    
    if dtype == 'single':
        recon = recon.view(c,h,w)
        mask = mask.view(h,w)
        splatted_wgts = splatted_wgts.view(h,w)

    if dtype == 'multi':
        recon = recon.view(B,D,c,h,w)
        mask = mask.view(B,D,1,h,w)
        splatted_wgts = splatted_wgts.view(B,D,1,h,w)
        if order == 'cdhw':
            recon = recon.permute(0,2,1,3,4)
            mask = mask.permute(0,2,1,3,4)
            splatted_wgts = splatted_wgts.permute(0,2,1,3,4)

    u = (xy_grid[...,0]-(w-1)/2)/w*(2*np.pi)
    v = -(xy_grid[...,1]-(h-1)/2)/h*np.pi
    uv = spherical_downsample(torch.stack([u,v],dim=1),downscale)

    return recon, mask, uv, coords, splatted_wgts

def uniform_depth_to_ply(depth,rgb,mat=None,basel=0.,eye='center',mask=None,coords=None,text=False): 

    assert eye in ['center','left','right']

    xyz = uniform_depth_to_xyz(depth,basel,eye)
    if not coords is None:
        xyz = torch.einsum('ij,jkl->ikl',coords.to(xyz.device),xyz)

    if not mat is None:
        ones = torch.ones_like(depth).unsqueeze(0)
        xyzone = torch.cat([xyz,ones],dim=0)
        imat = torch.linalg.inv(mat)
        xyzone = torch.einsum('ij,jkl->ikl',imat.to(xyzone.device),xyzone)
        xyz = xyzone[:3]

    if xyz.is_cuda:
        xyz = xyz.cpu()
    if rgb.is_cuda:
        rgb = rgb.cpu()
    rgb = (rgb*256).int().clamp(0,255)
    if not mask is None:
        x = xyz[0][mask].numpy()
        y = xyz[1][mask].numpy()
        z = xyz[2][mask].numpy()
        r = rgb[0][mask].numpy()
        g = rgb[1][mask].numpy()
        b = rgb[2][mask].numpy()
    else:
        x = xyz[0].view(-1).numpy()
        y = xyz[1].view(-1).numpy()
        z = xyz[2].view(-1).numpy()
        r = rgb[0].view(-1).numpy()
        g = rgb[1].view(-1).numpy()
        b = rgb[2].view(-1).numpy()
    dtype = [('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')]
    xyz = np.array(list(zip(x,y,z,r,g,b)),dtype=dtype)
    points = PlyElement.describe(xyz,'vertex')
    ply = PlyData([points],text=text)
    return ply # save plyfile by "ply.write(path)"
