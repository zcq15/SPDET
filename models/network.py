import sys
sys.path.append('.')
from .gargs import _args

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import os
import shutil
from os.path import join
from collections import OrderedDict
import numpy as np
import random
import cv2

import time
from tqdm import tqdm,trange

from .utils import tensor2img,depth2ply
from .geometry.rendering import scatter,inter
from .modules import spherical_interpolate


from .spdet.spdet import SPDET

# from .thirdparty.svsyn.svsyn import SvSyn
# from .thirdparty.selfnet360.selfnet360 import SelfNet360
# from .thirdparty.padenet.padenet import PADENet
# from .thirdparty.hohonet.hohonet import HoHoNet
# from .thirdparty.unifuse.unifuse import UniFuse
# from .thirdparty.omnifusion.omnifusion_iterative import OmniFusion
# from .thirdparty.joint360depth.joint360depth import Joint360Depth


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
            _input, _output = self.project(_input,_output,proj=_args['optim']['supervise'],filter=_args['optim']['filter'])
            for key in weights:
                metric_dict[key] = self.metrics[key](_output,_input)
        else:
            with torch.no_grad():
                _input, _output = self.net(_input,mode=mode,**kwargs)
                if _args['optim']['align']: _output = self.align(_input,_output)
                for key in weights:
                    metric_dict[key] = self.metrics[key](_output,_input)
        torch.distributed.barrier()
        return _input, _output, metric_dict

    def align(self,_input,_output):
        if not 'src_depth' in _input:
            max_values = _output['src_depth'].max(dim=-1,keepdim=True).values.max(dim=-2,keepdim=True).values
            _output['src_depth'] = _output['src_depth']/max_values*_args['data']['max']
            return _output
        if 'src_valid' in _input:
            mask = _input['src_valid']
        else:
            mask = torch.ones_like(_input['src_depth'],dtype=torch.bool)
        
        in_median = []
        out_median = []
        for b in range(_input['src_depth'].shape[0]):
            y = _input['src_depth'][b][mask[b]].median()
            x = _output['src_depth'][b][mask[b]].median()
            in_median.append(y)
            out_median.append(x)
        
        in_median = torch.stack(in_median).view([_input['src_depth'].shape[0],1,1,1])
        out_median = torch.stack(out_median).view([_input['src_depth'].shape[0],1,1,1])

        _output['src_depth'] = _output['src_depth']*in_median/out_median

        return _output

    def infer(self,_input,mode,weights,**kwargs):
        _input, _output = self.net(_input,mode=mode,**kwargs)
        return _input,_output

    def project(self,_input,_output,proj='scatter',filter=True,**kwargs):
        if proj == 'scatter':
            source, src_depth, tar_pos = _input['source'], _output['src_depth'], _input['tar_pos']
            tar_rot = None if not 'tar_rot' in _input else _input['tar_rot']
            src_pigment = torch.ones_like(src_depth,dtype=torch.bool)
            if filter and getattr(self.pars,'epoch',float('inf')) >= _args['optim']['pmwarmup']:
                with torch.no_grad():
                    _,_ans= self.net(_input,key='tar_rgb',mode='test',**kwargs)
                    tar_depth = _ans['src_depth']
                    inv_pos = -tar_pos
                    rot_inv = None if tar_rot is None else -tar_rot
                    _,src_scatter,_,_,_ = scatter(_input['target'],tar_depth,inv_pos,mask=(tar_depth>0)*(tar_depth<2*_args['data']['max']),\
                        upscale=_args['optim']['upscale'],downscale=_args['optim']['downscale'],th=_args['data']['max'],\
                        rot=rot_inv,rot_first=True)
                if src_scatter.type(torch.float32).mean() > _args['optim']['pmth'] * src_pigment.type(torch.float32).mean():
                    src_pigment = spherical_interpolate(src_scatter.type(torch.float32),_args['optim']['downscale'],_args['optim']['upscale']) > 0.5
    
            _output['tar_rgb_scatter'],_output['tar_valid_scatter'],_output['tar_uv_scatter'],_,_ = scatter(source,src_depth,tar_pos,\
                mask=src_pigment,upscale=_args['optim']['upscale'],downscale=_args['optim']['downscale'],\
                th=_args['data']['max'],rot=tar_rot,rot_first=False)

        elif proj == 'inter':
            if hasattr(self,'posenet'):
                target, src_depth, tar_pos = _input['target'], _output['src_depth'], _output['tar_pos']
                tar_rot = None if not 'tar_rot' in _output else _output['tar_rot']
            else:
                target, src_depth, tar_pos = _input['target'], _output['src_depth'], _input['tar_pos']
                tar_rot = None if not 'tar_rot' in _input else _input['tar_rot']
            _output['src_rgb_inter'] = inter(target,src_depth,tar_pos,rot=tar_rot,rot_first=False)

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
        total_num = 0
        total_num += sum(p.numel() for p in self.core.net.parameters())
        trainable_num = 0
        trainable_num += sum(p.numel() for p in self.core.net.parameters() if p.requires_grad)
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

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min',factor=_args['optim']['lr_decay'], patience=_args['optim']['lr_patience'],min_lr=_args['optim']['lr_min'],verbose=self.rank==0)

        ckpt = os.path.join(self.checkpoints,_args['arch']['model'].lower()+"-epoch-{:04d}.ckpt".format(_args['logs']['load_epoch']))\
                if _args['logs']['load_epoch'] >= 0 else os.path.join(self.checkpoints,_args['arch']['model'].lower()+"-latest.pt")
        if (_args['logs']['reload'] or not _args['arch']['mode'] == 'train'):
            self.load_ckpt(ckpt,ddp=True)
        if _args['logs']['reset_epoch']:
            self.epoch = 0

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

    def fps(self,iters=1000):
        self.core.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            _input = {'src_rgb':torch.randn([1,3,_args['data']['imgsize'][0],_args['data']['imgsize'][1]],dtype=torch.float32).cuda()}
            
            _ans = self.core.module.net(_input,mode='test')
            riters = trange(iters) if self.rank == 0 else range(iters)
            time_begin = time.time()
            for i in riters:
                _ans = self.core.module.net(_input,mode='test')
            time_end = time.time()
            if self.rank == 0: print('FPS: {:.3f} ms/f, {:.3f} f/s'.format((time_end-time_begin)/iters*1000,iters/(time_end-time_begin)))
        torch.cuda.empty_cache()

    def save_ckpt(self,ckpt):
        state_dict = {}
        state_dict['model'] = self.core.module.net.state_dict()
        state_dict['optim'] = self.optimizer.state_dict()
        state_dict['epoch'] = self.epoch
        state_dict['iter'] = self.iter
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
            if 'optim' in state_dict and hasattr(self,'optimizer') and _args['optim']['load_optim']: self.optimizer.load_state_dict(state_dict['optim'])
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
        # if _args['optim']['clip']:
        #     if dtype in ['rgb','depth']:
        #         for key in keys:
        #             if key in self._input:
        #                 H = self._input[key].shape[-2]
        #                 self._input[key] = self._input[key][...,H//4:-H//4,:]
        #             if key in self._output:
        #                 H = self._output[key].shape[-2]
        #                 self._output[key] = self._output[key][...,H//4:-H//4,:]

        assert dtype in ['rgb','depth','error','points','gray','label','bool','raw','npy']
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
                        valid = self._input['src_valid'][b][0] if 'src_valid' in self._input else None
                        item = tensor2img(self._input[key][b,0],imtype='turbo',mask=valid)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_input.png'.format(key)),item)
                if key in self._output:
                    for b in range(B):
                        # visual = self._input['visual'][b] if 'visual' in self._input else None
                        valid = self._input['src_valid'][b][0] if 'src_valid' in self._input else None
                        item = tensor2img(self._output[key][b,0],imtype='turbo',mask=valid)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_output_{}.png'.format(key,suffix)),item)
        elif dtype == 'raw':
            for key in keys:
                if key in self._input:
                    for b in range(B):
                        valid = self._input['src_valid'][b][0] if 'src_valid' in self._input else None
                        item = tensor2img(self._input[key][b,0],imtype=np.uint16,mask=valid,scale=4000)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_rawdepth_input.png'.format(key)),item)
                if key in self._output:
                    for b in range(B):
                        # visual = self._input['visual'][b] if 'visual' in self._input else None
                        valid = self._input['src_valid'][b][0] if 'src_valid' in self._input else None
                        item = tensor2img(self._output[key][b,0],imtype=np.uint16,mask=valid,scale=4000)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_rawdepth_output_{}.png'.format(key,suffix)),item)
                if key in self._input:
                    for b in range(B):
                        valid = self._input['src_valid'][b][0] if 'src_valid' in self._input else None
                        disp = 1/self._input[key][b,0]
                        disp = torch.where(torch.isnan(disp)|torch.isinf(disp),torch.zeros_like(disp),disp)
                        item = tensor2img(disp,imtype=np.uint16,mask=valid,scale=4000)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_rawdisp_input.png'.format(key)),item)
                if key in self._output:
                    for b in range(B):
                        # visual = self._input['visual'][b] if 'visual' in self._input else None
                        valid = self._input['src_valid'][b][0] if 'src_valid' in self._input else None
                        disp = 1/self._output[key][b,0]
                        disp = torch.where(torch.isnan(disp)|torch.isinf(disp),torch.zeros_like(disp),disp)
                        item = tensor2img(disp,imtype=np.uint16,mask=valid,scale=4000)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_rawdisp_output_{}.png'.format(key,suffix)),item)
        elif dtype == 'npy':
            for key in keys:
                if key in self._input:
                    for b in range(B):
                        item = self._input[key][b].cpu().numpy()
                        np.save(join(basedir,areas[b],fns[b]+'_{}_input.npy'.format(key)),item)
                if key in self._output:
                    for b in range(B):
                        item = self._output[key][b].cpu().numpy()
                        np.save(join(basedir,areas[b],fns[b]+'_{}_output_{}.npy'.format(key,suffix)),item)
        elif dtype == 'error':
            for key in keys:
                if key in self._input and key in self._output:
                    for b in range(B):
                        if 'src_valid' in self._input:
                            valid = self._input['src_valid'][b,0]
                            if 'src_visual' in self._input: valid *= self._input['src_visual'][b]
                        else:
                            valid = None
                        error = (self._input[key][b,0]-self._output[key][b,0]).abs()/self._input[key][b,0]*10.0
                        error[torch.isinf(error)|torch.isnan(error)] = 0
                        item = tensor2img(error,imtype='turbo',mask=valid)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_error_{}.png'.format(key,suffix)),item)
        elif dtype == 'points':
            for key in keys:
                if key in self._input:
                    for b in range(B):
                        valid = self._input['src_valid'][b,0] if 'src_valid' in self._input else None
                        visual = self._input['src_visual'][b] if 'src_visual' in self._input else None
                        if not valid is None and not visual is None:
                            valid = valid * visual
                        ply = depth2ply(self._input[key][b,0],self._input['rgb'][b],mask=valid)
                        ply.write(join(basedir,areas[b],fns[b]+'_{}_input.ply'.format(key)))
                if key in self._output:
                    for b in range(B):
                        visual = self._input['src_visual'][b] if 'src_visual' in self._input else None
                        ply = depth2ply(self._output[key][b,0],self._input['rgb'][b],mask=visual)
                        ply.write(join(basedir,areas[b],fns[b]+'_{}_output_{}.ply'.format(key,suffix)))
        elif dtype == 'gray':
            for key in keys:
                if key in self._input:
                    for b in range(B):
                        item = tensor2img(self._input[key][b],imtype=np.uint8,scale=255.0)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_input.png'.format(key)),item)
                if key in self._output:
                    for b in range(B):
                        item = tensor2img(self._output[key][b],imtype=np.uint8,scale=255.0)
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_output_{}.png'.format(key,suffix)),item)
        elif dtype == 'label':
            for key in keys:
                if key in self._input:
                    for b in range(B):
                        item = tensor2img(self._input[key][b],imtype='label')
                        cv2.imwrite(join(basedir,areas[b],fns[b]+'_{}_input.png'.format(key)),item)
                if key in self._output:
                    for b in range(B):
                        item = tensor2img(self._output[key][b],imtype='label')
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
        elif tuple(_input.shape[1:]) in [(3,),(3,1,1),(6,),(6,1,1),(7,),(7,1,1),(3,4),(4,4)]:
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
        self.save(bool_keys,'bool')

    def train(self,dataloader,weights,warmup=None,**kwargs):
        assert _args['arch']['mode'] == 'train'
        self.core.train()
        self.core.module.pars['epoch'] = self.epoch
        if _args['optim']['lr_update'] > 0:
            lr = max(_args['optim']['lr_min'],_args['optim']['lr']*_args['optim']['lr_decay']**(self.epoch//_args['optim']['lr_update']))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            if self.rank == 0: print("Learning rate: {:.3e}".format(lr))
        elif _args['optim']['lr_update'] == 0:
            lr = _args['optim']['lr']
            if self.rank == 0: print("Set with ReduceLROnPlateau, the initial value is {:.3e}".format(lr))
        else:
            lr = _args['optim']['lr']
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            if self.rank == 0: print("Learning rate: {:.3e}".format(lr))

        volumes = OrderedDict({'Size':0})
        for key in weights:
            volumes[key] = 0
        
        weights_valid = OrderedDict(weights)
        if not warmup is None:
            for key in warmup:
                if self.epoch < warmup[key]: del weights_valid[key]
        if self.rank == 0: print('valid weights: ',weights_valid)
        dataloader = tqdm(dataloader,dynamic_ncols=True) if self.rank == 0 else dataloader
        for idx, _input in enumerate(dataloader):
            self.set_input(_input)
            self.optimizer.zero_grad()

            _input, _output, loss_dict = self.core(self._input,_args['arch']['mode'],weights_valid,**kwargs)
            
            self._input = _input
            self.set_output(_output)
            loss = 0
            msg = 'Epoch {}/{} |'.format(self.epoch+1,_args['optim']['epochs'])
            volumes['Size'] += _input['src_rgb'].shape[0]
            for key in loss_dict:
                loss = loss + loss_dict[key]*weights_valid[key]
                value = reduce_tensor(loss_dict[key]).item()
                volumes[key] += value*_input['src_rgb'].shape[0]
                msg += '' if value == 0 else ' {}:{:.3f} |'.format(key,value)
                if self.rank == 0: self.writer.add_scalar("TRAIN/"+key,loss_dict[key].detach().item(),self.iter)

            if self.rank == 0: self.writer.flush()

            msg += ' Overall ::'
            for key in loss_dict:
                msg += '' if volumes[key] == 0 else ' {}:{:.3f} |'.format(key,volumes[key]/volumes['Size'])
                if self.rank == 0: self.writer.add_scalar("TRAIN/Overall "+key,volumes[key]/volumes['Size'],self.iter)
            if self.rank == 0: dataloader.set_postfix_str(msg)

            loss.backward()
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
                self.save_ckpt(os.path.join(self.checkpoints,"{}-epoch-{:04}.ckpt".format(_args['arch']['model'].lower(),self.epoch)))

    def test(self,dataloader,metrics,**kwargs):
        assert _args['arch']['mode'] in ['val','test']
        self.core.eval()
        volumes = OrderedDict({'Size':0})
        for key in metrics:
            volumes[key] = 0

        with torch.no_grad():
            dataloader = tqdm(dataloader,dynamic_ncols=True) if self.rank == 0 else dataloader
            for idx, _input in enumerate(dataloader):
                self.set_input(_input)
                _input, _output, loss_dict = self.core(self._input,_args['arch']['mode'],metrics,**kwargs)
                self._input = _input
                self.set_output(_output)
                volumes['Size'] += _input['src_rgb'].shape[0]
                for key in metrics:
                    volumes[key] += reduce_tensor(loss_dict[key]).item()*_input['src_rgb'].shape[0]
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
                self.writer.add_scalar("TEST/"+key,volumes[key]/volumes['Size'],self.epoch)
            self.writer.flush()
        return volumes
