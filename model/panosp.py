import torch
from torch import nn
from torch.nn import functional as F

from utils.layers import _make_pad, _make_norm, _make_act

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=[1,1], norm='bn', act='relu'):
        modules = [
            _make_pad(dilation),
            nn.Conv2d(in_channels, out_channels, 3, dilation=dilation, bias = norm is None),
            _make_norm(norm,out_channels),
            _make_act(act)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu', **kwargs):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias = norm is None),
            _make_norm(norm,out_channels),
            _make_act(act)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ERPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu'):
        super().__init__()
        self.horizontal = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias = norm is None),
            _make_norm(norm,out_channels),
            _make_act(act)
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias = norm is None),
            _make_norm(norm,out_channels),
            _make_act(act)
        )

    def forward(self,x):
        b,c,h,w = x.shape
        vector_h = x.mean(dim=2,keepdim=True)
        vector_h = self.horizontal(vector_h)
        vector_v = x.mean(dim=3,keepdim=True)
        vector_v = self.vertical(vector_v)

        feat = vector_h+vector_v
        return feat

class Squeeze(nn.Module):
    def __init__(self, in_channels, out_channels=None, strict=True, norm='bn', act='relu'):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        if out_channels == in_channels and not strict:
            self.layer = nn.Sequential()
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias = norm is None),
                _make_norm(norm,out_channels),
                _make_act(act)
            )
    
    def forward(self,x):
        return self.layer(x)

class Project(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm='bn', act=None):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, 1, bias = norm is None))
        modules.append(_make_norm(norm,out_channels))
        if act: modules.append(_make_act(act))
        self.layer = nn.Sequential(*modules)
    
    def forward(self,x):
        return self.layer(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channel = max(channel // reduction, 32)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=None, atrous_rates=[12, 24, 36]):
        super(ASPP, self).__init__()
        if out_channels is None:
            out_channels = in_channels//2

        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class PanoSP(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm='bn', act='relu'):
        super().__init__()
        if out_channels is None: out_channels = in_channels

        self.squeeze = Project(in_channels, out_channels, norm=norm, act=act)

        modules = []
        modules.append(ASPPPooling(in_channels, out_channels, norm='none', act=act))
        modules.append(ERPPooling(in_channels, out_channels, norm=norm, act=act))
        modules.append(Squeeze(in_channels, out_channels, norm=norm, act=act))
        self.convs = nn.ModuleList(modules)

        self.selayer = SELayer(len(self.convs) * out_channels)

        self.project = Project(len(self.convs) * out_channels, out_channels, norm=norm, act=act)

        self.relu = _make_act(act)

    def forward(self, x):
        y = self.squeeze(x)
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.selayer(res)
        res = self.project(res)
        return self.relu(y+res)
