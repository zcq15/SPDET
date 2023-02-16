import torch 
import numpy as np
from packaging.version import Version

def xy_grid(h,w,dim=0):
    if Version(torch.__version__) >= Version('1.10.0'):
        y,x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    else:
        y,x = torch.meshgrid(torch.arange(h), torch.arange(w))
    return torch.stack([x.cuda(),y.cuda()],dim=dim)

def uv_grid(h,w,dim=0):
    '''
    u -> (-pi,   pi),   i.e. theta
    v ^  (-pi/2, pi/2), i.e. phi

    u: -pi ---> pi
           pi/2
             ^
    v:      /|\ 
             |
           -pi/2
    '''
    if Version(torch.__version__) >= Version('1.10.0'):
        y,x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    else:
        y,x = torch.meshgrid(torch.arange(h), torch.arange(w))
    u = (x.cuda().type(torch.float32)-(w-1)/2)/w*(2*np.pi)
    v = -(y.cuda().type(torch.float32)-(h-1)/2)/h*np.pi
    return torch.stack([u,v],dim=dim)

def xyz_grid(h,w,dim=0):
    '''
          z
          |
          |
          |_ _ _ _ y
         /
        /
       x
    '''
    if Version(torch.__version__) >= Version('1.10.0'):
        y,x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    else:
        y,x = torch.meshgrid(torch.arange(h), torch.arange(w))
    u = (x.cuda()-(w-1)/2)/w*(2*np.pi)
    v = -(y.cuda()-(h-1)/2)/h*np.pi
    x = -torch.cos(v)*torch.cos(u)
    y = torch.cos(v)*torch.sin(u)
    z = torch.sin(v)
    return torch.stack([x,y,z],dim=dim)

def sp2car(depth):
    if len(depth.shape) == 3:
        b,h,w = depth.shape
    else:
        assert len(depth.shape) == 4 and depth.shape[1] == 1
        b,_,h,w = depth.shape
    assert h*2 == w
    xyz = xyz_grid(h,w).unsqueeze(0).expand(b,-1,-1,-1)
    return torch.clip(depth,0).view(b,1,h,w)*xyz

def car2sp(x,y,z,dim=0):
    r = torch.norm(torch.stack([x,y,z],dim=-1),p=2,dim=-1)
    u = torch.atan2(y,-x)
    v = torch.atan2(z,torch.norm(torch.stack([x,y],dim=-1),p=2,dim=-1))
    return torch.stack([r,u,v],dim=dim)

def uv2xy(u,v,h,w,dim=0):
    assert h*2 == w
    x = u/(2*np.pi)*w+(w-1)/2
    y = -v/np.pi*h+(h-1)/2

    x = torch.remainder(x, w)
    y = torch.remainder(y, h)

    return torch.stack([x,y],dim=dim)

def spattn(h,w,axis='z'):
    assert axis in ['x','y','z']
    uv = uv_grid(h,w).cuda()
    if axis == 'z':
        attn = torch.cos(uv[1])
    elif axis == 'y':
        attn = torch.cos(uv[1])*torch.cos(uv[0]).abs()
    elif axis == 'x':
        attn = torch.cos(uv[1])*torch.sin(uv[0]).abs()
    return attn
