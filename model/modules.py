import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')
sys.path.append('..')

from functools import partial
import timm
from utils.layers import _make_pad,_make_act,_make_norm
from utils.layers import Upsample,OFRS
from .vit import ViTEncoder
from .panosp import PanoSP
from .vgg import VggEncoder
from utils.geometry import uv_grid,xyz_grid


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1).contiguous()
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Readout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super().__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)
        return self.project(features)

class ResidualLayer(nn.Module):
    def __init__(self, features, norm=None, act='gelu', forward='refinenet'):
        super().__init__()
        assert forward in ['resnetv1','resnetv2','refinenet']
        self.forward = getattr(self,'forward_{}'.format(forward))
        self.pad = _make_pad(1)
        self.conv1 = nn.Conv2d(features,features,kernel_size=3,stride=1,padding=0,bias=not norm,groups=1)
        self.norm1 = _make_norm(norm, features)
        self.conv2 = nn.Conv2d(features,features,kernel_size=3,stride=1,padding=0,bias=not norm,groups=1)
        self.norm2 = _make_norm(norm, features)
        self.act = _make_act(act)

    def forward_resnetv1(self, x):
        out = x
        out = self.conv1(self.pad(out))
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(self.pad(out))
        out = self.norm2(out)
        return self.act(x+out)

    def forward_resnetv2(self, x):
        out = x
        out = self.act(out)
        out = self.norm1(out)
        out = self.conv1(self.pad(out))
        out = self.act(out)
        out = self.norm2(out)
        out = self.conv2(self.pad(out))
        return out+x

    def forward_refinenet(self, x):
        out = x
        out = self.act(out)
        out = self.conv1(self.pad(out))
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(self.pad(out))
        out = self.norm2(out)
        return out+x

class Aggregate(nn.Module):
    def __init__(self,features,upsample='ofrs',upnorm=None,upact='gelu',resnorm=None,resact='gelu',forward='refinenet',expand=False,align_corners=False,**kargs):
        super().__init__()

        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand == True: out_features = features // 2

        if upsample == 'ofrs':
            self.up = OFRS(features, out_features, ofrs_grid='geo', norm=upnorm, act=upact, force=False)
        elif upsample == 'inter':
            self.up = Upsample(features, out_features, norm=upnorm, act=upact)
        else:
            raise KeyError('Unexpected keys!')

        self.res1 = ResidualLayer(features, norm=resnorm, act=resact, forward=forward)
        self.res2 = ResidualLayer(features, norm=resnorm, act=resact, forward=forward)

        # self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.res1(xs[1])
            output = output + res

        output = self.res2(output)

        output = self.up(output)

        return output


class Embed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, vgg='vgg19', vgg_grid='geo', vgg_hooks=1, vgg_embed='mlp', input_size=(512,1024), patch_size=16, shape_embed=(32,64), num_tokens=1, **kwargs):
        super().__init__()

        self.input_size = tuple(input_size)
        self.out_channels = [256, 512, 1024]

        pretrained_model = ViTEncoder(in_channels=in_channels)

        self.layer0 = pretrained_model.layer0
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3

        del pretrained_model

        if vgg_grid in ['cos','sin','xyz']:
            vgg_channels = 3
        elif vgg_grid in ['uv']:
            vgg_channels = 2
        elif vgg_grid in ['geo','disctn']:
            vgg_channels = 5
        elif vgg_grid in ['none',None]:
            vgg_channels = 0
        else:
            raise KeyError

        self.vgg = None
        self.vgg_grid = None
        if not vgg_grid in ['none',None] and vgg_hooks > 0:
            self.vgg = VggEncoder(vgg=vgg,in_channels=vgg_channels,vgg_hooks=vgg_hooks)

        assert input_size[0] % patch_size == 0 and input_size[1] % patch_size == 0
        self.patch_size = patch_size
        self.shape_embed = tuple(shape_embed)
        self.num_patches = self.shape_embed[0] * self.shape_embed[1]
        self.num_tokens = num_tokens

        if vgg_grid in ['none',None] or vgg_hooks <= 0:
            self.project = nn.Conv2d(self.out_channels[-1], embed_dim, kernel_size=1)
        else:
            for l in range(4-len(self.vgg.out_channels),4):
                in_channels = self.out_channels[-(4-l)]+self.vgg.out_channels[-(4-l)]
                out_channels = embed_dim if l == 3 else self.out_channels[-(4-l)]
                if vgg_embed == 'linear':
                    embed = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                elif vgg_embed == 'mlps':
                    embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True),
                        # nn.GELU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True),
                    )
                else:
                    embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True)
                    )
                self.add_module('embed'+str(l),embed)
            self.vgg_feats = None

        self.token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens+self.num_patches, embed_dim))
        self.pos_drop = nn.Identity() #nn.Dropout(p=0.)

        self.register_embed = partial(self._register_embed,vgg_grid=vgg_grid)
        torch.cuda.empty_cache()

    def _register_embed(self,shape,vgg_grid='geo'):
        b,_,h,w = shape
        if vgg_grid == 'cos':
            uv = uv_grid(h,w).view(1,2,h,w)
            grid = torch.cat([torch.cos(uv[:,:1]),torch.cos(uv[:,1:]),torch.cos(uv[:,:1])*torch.cos(uv[:,1:])],dim=1)
            self.vgg_grid = grid.expand([b,-1,-1,-1])
        elif vgg_grid == 'sin':
            uv = uv_grid(h,w).view(1,2,h,w)
            grid = torch.cat([torch.sin(uv[:,:1]),torch.sin(uv[:,1:]),torch.sin(uv[:,:1])*torch.sin(uv[:,1:])],dim=1)
            self.vgg_grid = grid.expand([b,-1,-1,-1])
        elif vgg_grid == 'xyz':
            xyz = xyz_grid(h,w,dim=0).view(1,3,h,w)
            self.vgg_grid = xyz.expand([b,-1,-1,-1])
        elif vgg_grid == 'uv':
            uv = uv_grid(h,w).view(1,2,h,w)
            self.vgg_grid = uv.expand([b,-1,-1,-1])
        elif vgg_grid == 'geo':
            uv = uv_grid(h,w).view(1,2,h,w)
            grid = torch.cat([torch.sin(uv[:,:1]),torch.cos(uv[:,:1]),torch.sin(uv[:,1:]),torch.cos(uv[:,1:]),torch.cos(uv[:,:1])*torch.cos(uv[:,1:])],dim=1)
            self.vgg_grid = grid.expand([b,-1,-1,-1])
        elif vgg_grid == 'disctn':
            uv = uv_grid(h,w).view(1,2,h,w)
            grid = torch.cat([torch.sin(uv[:,:1]),torch.cos(uv[:,:1]),torch.sin(uv[:,1:]),torch.cos(uv[:,1:]),torch.sin(uv[:,:1])*torch.sin(uv[:,1:])],dim=1)
            self.vgg_grid = grid.expand([b,-1,-1,-1])
        else:
            raise KeyError

    def forward(self, inputs):
        if not self.vgg is None:

            if self.vgg_grid is None \
                or not self.vgg_grid.shape[0] == inputs.shape[0] \
                or not tuple(self.vgg_grid.shape[-2:]) == tuple(inputs.shape[-2:]):
                self.register_embed(inputs.shape)
                self.vgg_feats = None

            if self.training:
                self.vgg_feats = None
                feats = self.vgg(self.vgg_grid)

            elif self.vgg_feats is None:
                self.vgg_feats = self.vgg(self.vgg_grid)
                feats = self.vgg_feats
    
            else:
                feats = self.vgg_feats


        x = inputs
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        if hasattr(self,'embed1'):
            x1 = self.embed1(torch.cat([x1,feats[-3]],dim=1))
        x2 = self.layer2(x1)
        if hasattr(self,'embed2'):
            x2 = self.embed2(torch.cat([x2,feats[-2]],dim=1))
        x3 = self.layer3(x2)
        if hasattr(self,'embed3'):
            x3 = self.embed3(torch.cat([x3,feats[-1]],dim=1))
        if hasattr(self,'project'):
            x3 = self.project(x3)
        x3 = x3.flatten(2).transpose(1, 2)

        pos_embed_token = self.pos_embed[:,:self.num_tokens]
        pos_embed_grid = self.pos_embed[:,self.num_tokens:]
        pos_embed_shape = (self.input_size[0]//self.patch_size, self.input_size[1]//self.patch_size)
        if not tuple(self.shape_embed) == tuple(pos_embed_shape):
            pos_embed_grid = pos_embed_grid.transpose(1, 2).unflatten(2,self.shape_embed)
            pos_embed_grid = F.interpolate(pos_embed_grid, size=pos_embed_shape, mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)

        pos_embed = torch.cat([pos_embed_token,pos_embed_grid],dim=1)

        b,_,_,_ = x.shape
        x3 = torch.cat([self.token.expand(b,-1,-1),x3],dim=1)
        x3 = x3 + pos_embed
        x3 = self.pos_drop(x3)

        return [x1,x2],x3

class Transformer(nn.Module):
    def __init__(self, layers=12, hooks=[8,11], embed_dim=768, num_heads=12, **kargs):
        super().__init__()
        if layers == 12 and embed_dim == 768 and num_heads == 12:
            self.layers = timm.create_model('vit_base_r50_s16_384',pretrained=True).blocks
        else:
            self.layers = nn.ModuleList([TransformerLayer(embed_dim,num_heads,**kargs) for _ in range(layers)])
        self.hooks = hooks
        torch.cuda.empty_cache()

    def forward(self,x):
        _output = []
        for l in range(len(self.layers)):
            x = self.layers[l](x)
            if l in self.hooks: _output.append(x)
        return _output

class Reassemble(nn.Module):
    def __init__(self, input_size=(512,1024), channels=[256, 512, 768, 768], panosp=True, embed_dim=768, patch_size=16, num_tokens=1):
        super().__init__()
        self.input_size = input_size
        self.channels = channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.layer1 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
        self.layer2 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
        self.layer3 = nn.Sequential(
            Readout(embed_dim,num_tokens),
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([input_size[0]//patch_size, input_size[1]//patch_size])),
            nn.Conv2d(embed_dim,channels[2],kernel_size=1)
        )

        self.layer4 = nn.Sequential(
            Readout(embed_dim,num_tokens),
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([input_size[0]//patch_size, input_size[1]//patch_size])),
            PanoSP(embed_dim,embed_dim,norm='bn',act='relu') if panosp else nn.Identity(),
            nn.Conv2d(embed_dim,channels[3],kernel_size=1),
            _make_pad(1),
            nn.Conv2d(channels[3],channels[3],kernel_size=3,stride=2,padding=0)
        )

    def forward(self,_input):
        assert len(_input) == 4
        for l in range(4):
            _input[l] = getattr(self,'layer{}'.format(l+1))(_input[l])
        return _input

class Fusion(nn.Module):
    def __init__(self,in_channels=[256, 512, 768, 768],out_channels=256,upsample='ofrs',expand=False,align_corners=False):
        super().__init__()
        if expand:
            out_channels = [out_channels,out_channels*2,out_channels*4,out_channels*8]
        else:
            out_channels = [out_channels,out_channels,out_channels,out_channels]

        for l in range(len(in_channels)):
            layer = nn.Sequential(
                _make_pad(1),
                nn.Conv2d(in_channels[l],out_channels[l],kernel_size=3,stride=1,padding=0,bias=False)
            )
            self.add_module('layer{}'.format(l+1),layer)

        for l in range(len(out_channels)):
            fusion = Aggregate(out_channels[l],
                               upsample=upsample,upnorm=None,upact='gelu',
                               resnorm=None,resact='gelu',forward='refinenet',
                               expand=expand,align_corners=align_corners)
            self.add_module('fusion{}'.format(l+1),fusion)
        del self.fusion4.res1

    def forward(self,_input):
        assert len(_input) == 4

        for l in range(4):
            _input[l] = getattr(self,'layer{}'.format(l+1))(_input[l])

        x = self.fusion4(_input[3])
        x = self.fusion3(x,_input[2])
        x = self.fusion2(x,_input[1])
        x = self.fusion1(x,_input[0])

        return x

class Head(nn.Module):
    def __init__(self,channels=256, non_negative=True, **kargs):
        super().__init__()

        self.head = nn.Sequential(
            _make_pad(1),
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _make_pad(1),
            nn.Conv2d(channels // 2, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity()
        )

    def forward(self,x):
        depth = self.head(x)
        return depth
