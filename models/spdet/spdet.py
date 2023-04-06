import sys
sys.path.append('.')
sys.path.append('..')
from ..gargs import _args

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from collections import OrderedDict
from functools import reduce
from .modules import Embed, Transformer, Reassemble, Fusion, Head

class SPDET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, imgsize=None, embed_dim=768, fusion_dim=256, **kargs):
        super().__init__()

        self.register_buffer('x_mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]))
        self.register_buffer('x_std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]))

        if imgsize is None: imgsize = _args['data']['imgsize']

        self.embed = Embed(in_channels=in_channels, input_size=imgsize, embed_dim=embed_dim, patch_size=16, shape_embed=(imgsize[0]//16,imgsize[1]//16))

        self.transformer = Transformer(layers=12, hooks=[8,11], embed_dim=768, num_heads=12)

        self.reassemble = Reassemble(input_size=imgsize, channels=[256, 512, 768, 768], embed_dim=768, patch_size=16)

        self.fusion = Fusion(in_channels=[256, 512, 768, 768], out_channels=fusion_dim)

        self.head = Head(channels=fusion_dim, out_channels=out_channels, end=_args['data']['max'], **kargs)

        self.set_parameters(pars='all')
        torch.cuda.empty_cache()

    def forward(self, _input, key='src_rgb', mode='test', **kargs):

        if isinstance(_input,OrderedDict) or isinstance(_input,dict):
            equi = (_input[key] - self.x_mean) / self.x_std
        elif torch.is_tensor(_input):
            equi = _input
        else:
            raise KeyError

        _tokens, token = self.embed(equi)

        _tokens = _tokens + self.transformer(token)

        _tokens = self.reassemble(_tokens)

        _tokens = self.fusion(_tokens)

        predict = self.head(_tokens)

        if torch.is_tensor(_input): return predict

        _output = OrderedDict()
        _output['src_depth'] = predict

        # print([equi.shape, predict.shape])

        return _input,_output

    def load_state_dict(self, state_dict, **kwards):
        state_dict = OrderedDict(state_dict)
        _state_dict = self.state_dict()
        _update_dict = []
        freeze_keys = []
        for key in state_dict:
            if key in _state_dict:
                if state_dict[key].shape == _state_dict[key].shape:
                    freeze_keys.append(key)
                    _update_dict.append((key,state_dict[key].to(_state_dict[key].device)))
        _state_dict.update(_update_dict)
        del state_dict,_update_dict
        return nn.Module.load_state_dict(self,_state_dict)

    def predict(self, _input, **kwards):
        with torch.no_grad():
            return self.forward(_input, mode='test', **kwards)

    def set_parameters(self,pars='all'):
        if pars == 'all':
            for par in self.parameters():
                par.requires_grad = True
