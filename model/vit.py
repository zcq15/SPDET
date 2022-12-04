import torch
import torch.nn as nn

import timm
from utils.layers import _make_pad
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
                if m.same_pad:
                    assert m.padding ==(0,0)
                else:
                    assert not m.padding ==(0,0)
                m.same_pad = False
                m.padding = (0,0)
                pad = _make_pad((m.kernel_size[0]//2,m.kernel_size[1]//2))
                conv = nn.Sequential(pad,m)
                _set_module(model,n,conv)
    return model


class ViTEncoder(nn.Module):
    def __init__(self,  in_channels=3, **kwargs):
        super().__init__()

        pretrained_model = timm.create_model('vit_base_r50_s16_384',pretrained=True)
        if not in_channels == 3:
            pretrained_model.patch_embed.backbone._modules['stem'].conv = timm.models.layers.StdConv2dSame(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
        pretrained_model = _wrap_pad(pretrained_model)
        self.layer0 = pretrained_model.patch_embed.backbone._modules['stem']
        self.layer1 = pretrained_model.patch_embed.backbone._modules['stages'][0]
        self.layer2 = pretrained_model.patch_embed.backbone._modules['stages'][1]
        self.layer3 = pretrained_model.patch_embed.backbone._modules['stages'][2]

        del pretrained_model
        torch.cuda.empty_cache()

    def forward(self, inputs):
        # resnet
        x = inputs
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x5 = self.layre5(x4)
        # print([_.shape for _ in [x0,x1,x2,x3]])
        return [x0,x1,x2,x3]




