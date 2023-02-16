import torch
import torch.nn as nn

import timm
import sys
sys.path.append('.')
sys.path.append('..')
from ..modules import _make_pad
import copy

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def _wrap_pad(model):
    for n,m in copy.deepcopy(model).named_modules():
        if isinstance(m,nn.Conv2d):
            if not m.kernel_size == (1,1):
                pad = _make_pad((m.kernel_size[0]//2,m.kernel_size[1]//2))
                m.padding = (0,0)
                conv = nn.Sequential(pad,m)
                _set_module(model,n,conv)
    return model


class VggEncoder(nn.Module):
    def __init__(self,  vgg='vgg19', in_channels=3, vgg_hooks=1, **kwargs):
        super().__init__()
        out_channels = [256,512,512]
        if vgg == 'vgg19':
            pretrained_model = timm.create_model('vgg19',pretrained=True).features[:31]
            hooks = [12,21,30]
        else:
            pretrained_model = timm.create_model('vgg19_bn',pretrained=True).features[:44]
            hooks = [17,30,43]

        self.out_channels = out_channels[-vgg_hooks:]
        self.hooks = hooks[-vgg_hooks:]
        if not in_channels == 3:
            pretrained_model[0] = nn.Conv2d(in_channels,64,kernel_size=3,padding=1)
        pretrained_model = _wrap_pad(pretrained_model)
        self.vgg = pretrained_model

        del pretrained_model
        torch.cuda.empty_cache()

    def forward(self, inputs):
        # resnet
        x = inputs
        outputs = []
        for l in range(len(self.vgg)):
            x = self.vgg[l](x)
            if l in self.hooks:
                outputs += [x]
        return outputs



