import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict
from .modules import Embed, Transformer, Reassemble, Fusion, Head


class SPDET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, imgsize=[256, 512], embed_dim=768, fusion_dim=256, **kargs):
        super().__init__()

        self.register_buffer('x_mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]))
        self.register_buffer('x_std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]))

        self.embed = Embed(in_channels=in_channels, input_size=imgsize, embed_dim=embed_dim, patch_size=16, shape_embed=(imgsize[0]//16,imgsize[1]//16))

        self.transformer = Transformer(layers=12, hooks=[8,11], embed_dim=768, num_heads=12)

        self.reassemble = Reassemble(input_size=imgsize, channels=[256, 512, 768, 768], embed_dim=768, patch_size=16)

        self.fusion = Fusion(in_channels=[256, 512, 768, 768], out_channels=fusion_dim)

        self.head = Head(channels=fusion_dim, out_channels=out_channels, **kargs)

        torch.cuda.empty_cache()

    def forward(self, _input, **kargs):

        equi = (_input - self.x_mean) / self.x_std

        _tokens, token = self.embed(equi)

        _tokens = _tokens + self.transformer(token)

        _tokens = self.reassemble(_tokens)

        _tokens = self.fusion(_tokens)

        depth = self.head(_tokens)

        return depth

    def load_state_dict(self, state_dict, **kwards):
        state_dict = OrderedDict(state_dict)
        _state_dict = self.state_dict()
        _update_dict = []
        for key in state_dict:
            if key in _state_dict:
                if state_dict[key].shape == _state_dict[key].shape:
                    _update_dict.append((key,state_dict[key].to(_state_dict[key].device)))
        _state_dict.update(_update_dict)
        del state_dict,_update_dict
        return nn.Module.load_state_dict(self,_state_dict)

    def predict(self, _input, **kwards):
        with torch.no_grad():
            return self.forward(_input, mode='test', **kwards)

