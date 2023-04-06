import sys
sys.path.append('.')
from .gargs import _args

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import os
from os.path import join
from collections import OrderedDict
import numpy as np
import cv2
from tqdm import tqdm

from .utils import tensor2img
from .geometry.rendering import uniform_interpolate,uniform_scatter
# from .omninvs.omninvs import OmniNVS
from .spdet.spdet import SPDET

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    torch.distributed.barrier()
    rt = tensor.clone().detach()
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt = rt / dist.get_world_size()
    return rt

class Core(nn.Module):
    def __init__(self,metrics=None):
        super().__init__()

        ### initial
        self.net = eval(_args['arch']['model'])().cuda()
        self.metrics = nn.ModuleDict(metrics).cuda() if not metrics is None else {}
        self.pars = OrderedDict()

    def forward(self,_input,mode,weights,**kwargs):
        metric_dict = OrderedDict()
        if mode == 'train':
            _input, _output = self.net(_input,mode=mode,**kwargs)
            _input, _output = self.project(_input, _output, proj=_args['optim']['supervise'])
            for key in weights:
                metric_dict[key] = self.metrics[key](_output,_input)
        else:
            with torch.no_grad():
                _input, _output = self.net(_input,mode=mode,**kwargs)
                for key in weights:
                    metric_dict[key] = self.metrics[key](_output,_input)
        torch.distributed.barrier()
        return _input, _output, metric_dict

    def project(self,_input,_output,proj='scatter',**kwargs):
        if proj == 'scatter':
            source, src_depth, src_pos, tar_pos = _input['source'], _output['src_depth'], _input['src_pos'], _input['tar_pos']
            src_pigment = torch.ones_like(src_depth,dtype=torch.bool)
            with torch.no_grad():
                _,_ans= self.net(_input,key='tar_rgb',mode='test',**kwargs)
                tar_depth = _ans['src_depth']
                bwd_pos = src_pos - tar_pos
                _,src_scatter,_,_,_ = uniform_scatter(_input['target'],tar_depth,t=bwd_pos,mask=(tar_depth>0))
            if src_scatter.type(torch.float32).mean() > 0.5 * src_pigment.type(torch.float32).mean():
                src_pigment = src_scatter
            fwd_pos = tar_pos - src_pos
            _output['tar_rgb_scatter'],_output['tar_valid_scatter'],_output['tar_uv_scatter'],_,_ = uniform_scatter(source,src_depth,t=fwd_pos,mask=src_pigment)

        elif proj == 'inter':
            target, src_depth, src_pos, tar_pos = _input['target'], _output['src_depth'], _input['src_pos'], _input['tar_pos']
            fwd_pos = tar_pos - src_pos
            _output['src_rgb_inter'] = uniform_interpolate(target,src_depth,t=fwd_pos)

        else:
            pass

        return _input,_output

class Network(object):
    def __init__(self,metrics=None):
        super().__init__()

        ### initial
        self.rank = _args['arch']['rank']
        self.core = Core(metrics)
        if self.rank == 0 and _args['logs']['overview']: print(self.core.net)
        total_num = sum(p.numel() for p in self.core.net.parameters())
        trainable_num = sum(p.numel() for p in self.core.net.parameters() if p.requires_grad)
        if _args['arch']['rank']==0: print('Total pars: {:.3f}M, Trainable pars: {:.3f}M'.format(total_num/1e6,trainable_num/1e6))

        self._input = OrderedDict()
        self._output = OrderedDict()

        #### reload
        self.epoch = 0
        self.iter = 0
        self.logs = os.path.join(_args['logs']['logs'],_args['arch']['id'],_args['logs']['timestamp'])
        os.makedirs(self.logs,exist_ok=True)
        self.checkpoints = os.path.join(_args['logs']['checkpoints'],_args['arch']['id'])
        os.makedirs(self.checkpoints,exist_ok=True)

        self.core = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.core)
        self.core = torch.nn.parallel.DistributedDataParallel(self.core,device_ids=[self.rank],output_device=self.rank,find_unused_parameters=_args['optim']['unused'])

        ### initial for training
        if _args['arch']['mode'] == 'train':
            if _args['optim']['optim'] == 'Adam':
                self.optimizer = torch.optim.Adam(self.core.parameters(),lr=_args['optim']['lr'],weight_decay=_args['optim']['weight_decay'],betas=(_args['optim']['beta1'], _args['optim']['beta2']))
            else:
                self.optimizer = torch.optim.SGD(self.core.parameters(),lr=_args['optim']['lr'],weight_decay=_args['optim']['weight_decay'],momentum=0.9)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min',factor=0.5, patience=3,verbose=self.rank==0)

        ckpt = os.path.join(self.checkpoints,_args['arch']['model'].lower()+"-epoch-{:06d}.ckpt".format(_args['logs']['load_epoch']))\
                if _args['logs']['load_epoch'] >= 0 else os.path.join(self.checkpoints,_args['arch']['model'].lower()+"-latest.pt")
        if (_args['logs']['reload'] or not _args['arch']['mode'] == 'train'):
            self.load_ckpt(ckpt,ddp=True)
        if _args['logs']['reset_epoch']:
            self.epoch = 0
        del ckpt

        if _args['arch']['mode'] == 'train' and self.rank == 0:
            os.makedirs(self.logs,exist_ok=True)
            self.writer = SummaryWriter(self.logs)
            self.writer.add_text('CONFIG/FILE',_args['arch']['config'])
            self.writer.add_text('CONFIG/TEXT',_args['arch']['config_text'])
            self.writer.flush()
        else:
            self.writer = None
        if self.rank == 0:
            print("Use CUDNN Benchmark: ",torch.backends.cudnn.benchmark)
            print("Use Deterministic Algorithm: ",torch.backends.cudnn.deterministic)

        torch.cuda.empty_cache()

    def save_ckpt(self,ckpt):
        state_dict = {
            'model': self.core.module.net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iter': self.iter,
        }
        torch.save(state_dict,ckpt)

    def reset_optimizer(self):
        self.optimizer.state_dict()['state'] = {}

    def load_ckpt(self,ckpt,log=True,ddp=True):
        if self.rank == 0 and log: print('Load from {} ... '.format(ckpt))
        if os.path.exists(ckpt):
            state_dict = torch.load(ckpt,map_location='cpu')
            if ddp:
                self.core.module.net.load_state_dict(state_dict['model'],strict=True)
                self.core._sync_params_and_buffers()
            else:
                self.core.net.load_state_dict(state_dict['model'],strict=True)
            if 'optim' in state_dict and hasattr(self,'optimizer'): self.optimizer.load_state_dict(state_dict['optim'])
            if 'epoch' in state_dict: self.epoch = state_dict['epoch']
            if 'iter' in state_dict: self.iter = state_dict['iter']
            del state_dict
            torch.cuda.empty_cache()
            os.makedirs(self.logs,exist_ok=True)
            if self.rank == 0 and log: print('Load successfully!')
        else:
            if self.rank == 0 and log: print('No checkpoint found!')

    def set_input(self,_input):
        self._input = OrderedDict()
        for key in _input:
            self._input[key] = _input[key].cuda() if torch.is_tensor(_input[key]) and not _input[key].is_cuda else _input[key]

    def set_output(self,_output):
        self._output = OrderedDict()
        for key in _output.keys():
            if torch.is_tensor(_output[key]):
                self._output[key] = _output[key].detach()

    def save(self,keys,dtype):
        assert dtype in ['rgb','depth','bool']
        basedir = join(_args['logs']['savedir'],_args['arch']['dataset'])
        areas = self._input['area']
        fns = self._input['fn']
        B = len(self._input['fn'])
        for b in range(B):
            os.makedirs(join(basedir,areas[b]),exist_ok=True)
        suffix = _args['arch']['id'] if _args['arch']['suffix'] is None else '{}_{}'.format(_args['arch']['id'], _args['arch']['suffix'])
        if dtype == 'rgb':
            for key in keys:
                if key in self._input:
                    for b in range(B):
                        item = tensor2img(self._input[key][b],imtype=np.uint8,scale=255.0)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_input.png'.format(key)),item)
                if key in self._output:
                    for b in range(B):
                        item = tensor2img(self._output[key][b],imtype=np.uint8,scale=255.0)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_output_{}.png'.format(key,suffix)),item)
        elif dtype == 'depth':
            for key in keys:
                if key in self._input:
                    for b in range(B):
                        item = tensor2img(self._input[key][b,0],imtype='turbo')
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_input.png'.format(key)),item)
                if key in self._output:
                    for b in range(B):
                        item = tensor2img(self._output[key][b,0],imtype='turbo')
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_output_{}.png'.format(key,suffix)),item)
        elif dtype == 'bool':
            for key in keys:
                if key in self._input:
                    for b in range(B):
                        item = tensor2img(self._input[key][b],imtype='bool')
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_input.png'.format(key)),item)
                if key in self._output:
                    for b in range(B):
                        item = tensor2img(self._output[key][b],imtype='bool')
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_output_{}.png'.format(key,suffix)),item)

    def check_type(self,_input):
        if not torch.is_tensor(_input):
            return None
        elif len(_input.shape) < 3 or len(_input.shape) > 5 or _input.shape[-1] == 1:
            return None
        elif len(_input.shape) == 4 and _input.shape[1] == 3:
            return 'rgb'
        elif len(_input.shape) == 3 or _input.shape[1] == 1:
            if _input.dtype is torch.float32:
                return 'depth'
            elif _input.dtype is torch.bool:
                return 'bool'
            else:
                return None
        else:
            return None

    def auto_save(self):
        rgb_keys,depth_keys,bool_keys = set(),set(),set()
        for key in self._input:
            dtype = self.check_type(self._input[key])
            if dtype == 'rgb': rgb_keys.add(key)
            if dtype == 'depth': depth_keys.add(key)
            if dtype == 'bool': bool_keys.add(key)
        for key in self._output:
            dtype = self.check_type(self._output[key])
            if dtype == 'rgb': rgb_keys.add(key)
            if dtype == 'depth': depth_keys.add(key)
            if dtype == 'bool': bool_keys.add(key)
        self.save(rgb_keys,'rgb')
        self.save(depth_keys,'depth')
        # self.save(bool_keys,'bool')

    def train(self,dataloader,weights,warmup=None,leave=True,**kwargs):
        # torch.cuda.empty_cache()
        assert _args['arch']['mode'] == 'train'
        self.core.train()
        self.core.module.pars['epoch'] = self.epoch
        if _args['optim']['lr_update'] > 0:
            lr = max(_args['optim']['lr_min'],_args['optim']['lr']*_args['optim']['lr_decay']**(self.epoch//_args['optim']['lr_update']))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        elif _args['optim']['lr_update'] == 0:
            lr = _args['optim']['lr']
            pass
        else:
            lr = _args['optim']['lr']
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        # if self.rank == 0: print("Learning rate: {:.3e}".format(lr))

        volumes = OrderedDict({'Size':0})
        for key in weights:
            volumes[key] = 0
            # volumes_overall[key] = 0
        
        weights_valid = OrderedDict(weights)
        if not warmup is None:
            for key in warmup:
                if self.epoch < warmup[key]: del weights_valid[key]
        if self.rank == 0: print('weights: ',weights_valid)
        dataloader = tqdm(dataloader,dynamic_ncols=True,leave=leave) if self.rank == 0 else dataloader
        for idx, _input in enumerate(dataloader):
            self.set_input(_input)
            self.optimizer.zero_grad()
            _input, _output, loss_dict = self.core(self._input,_args['arch']['mode'],weights_valid,**kwargs)
            
            self._input = _input
            self.set_output(_output)
            # print(_output.keys())
            loss = 0
            msg = 'Epoch {}/{} |'.format(self.epoch+1,_args['optim']['epochs'])
            volumes['Size'] += _input['tar_rgb'].shape[0]
            for key in loss_dict:
                loss = loss + loss_dict[key]*weights_valid[key]
                value = reduce_tensor(loss_dict[key]).item()
                volumes[key] += value*_input['tar_rgb'].shape[0]
                msg += '' if value == 0 else ' {}:{:.3f} |'.format(key,value)
                if self.rank == 0: self.writer.add_scalar("TRAIN/"+key,loss_dict[key].detach().item(),self.iter)

            if self.rank == 0: self.writer.flush()

            msg += ' Overall ::'
            for key in loss_dict:
                msg += '' if volumes[key] == 0 else ' {}:{:.3f} |'.format(key,volumes[key]/volumes['Size'])
                if self.rank == 0: self.writer.add_scalar("TRAIN/Overall_"+key,volumes[key]/volumes['Size'],self.iter)
            if self.rank == 0: dataloader.set_postfix_str(msg)

            loss.backward()

            # if _args['optim']['grad_clip'] > 0:
            #     nn.utils.clip_grad_norm_(self.core.parameters(), max_norm=_args['optim']['grad_clip'])
            
            self.optimizer.step()

            self.iter += 1

            if _args['logs']['debug']:
                if _args['logs']['saveimg']:
                    self.auto_save()

        self.epoch += 1
        if _args['arch']['world_size'] > 1 : dist.barrier()
        if self.rank == 0:
            self.save_ckpt(os.path.join(self.checkpoints,_args['arch']['model'].lower()+"-latest.pt"))
            if self.epoch % _args['logs']['ckpt_update'] == 0:
                self.save_ckpt(os.path.join(self.checkpoints,"{}-epoch-{:06}.ckpt".format(_args['arch']['model'].lower(),self.epoch)))
        
        return volumes
        # torch.cuda.empty_cache()

    def test(self,dataloader,metrics,leave=True,**kwargs):
        assert _args['arch']['mode'] in ['val','test']
        self.core.eval()
        volumes = OrderedDict({'Size':0})
        for key in metrics:
            volumes[key] = 0

        with torch.no_grad():
            dataloader = tqdm(dataloader,dynamic_ncols=True,leave=leave) if self.rank == 0 else dataloader
            for idx, _input in enumerate(dataloader):
                self.set_input(_input)
                _input, _output, loss_dict = self.core(self._input,_args['arch']['mode'],metrics,**kwargs)
                self._input = _input
                self.set_output(_output)
                volumes['Size'] += _input['tar_rgb'].shape[0]
                for key in metrics:
                    volumes[key] += reduce_tensor(loss_dict[key]).item()*_input['tar_rgb'].shape[0]
                msg = 'Epoch {}/{} | Overall :: '.format(self.epoch,_args['optim']['epochs'])
                for key in loss_dict:
                    msg += '' if volumes[key] == 0 else ' {}:{:.4f} |'.format(key,volumes[key]/volumes['Size'])
                dataloader.set_postfix_str(msg) if self.rank == 0 else None

                if _args['logs']['saveimg']:
                    self.auto_save()

        if _args['optim']['lr_update'] == 0:
            self.scheduler.step(loss_dict[list(loss_dict.keys())[0]])

        if self.rank == 0 and not self.writer is None:
            for key in metrics:
                self.writer.add_scalar(_args['arch']['mode'].upper()+"/Mean "+key,volumes[key]/volumes['Size'],self.epoch)
            self.writer.flush()
        # torch.cuda.empty_cache()
        return volumes
