import argparse
import yaml
import os
import time
import datetime
import math
from collections import OrderedDict

import sys

sys.path.append('.')

from . import gargs

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None: 
            setattr(namespace, self.dest, None)
            return None
        setattr(namespace, self.dest, dict())
        for value in values:
            if ':' in value:
                key, value = value.strip().split(':')
                getattr(namespace, self.dest)[key] = float(value)

class Options(object):

    def _init_global(self):
        for key in self.args:
            gargs._args[key] = OrderedDict()

    def _set_global(self):
        for d in self.args:
            for k in self.args[d]:
                gargs._args[d][k] = getattr(self.opt,k)

    def __init__(self):
        # create ArgumentParser() obj
        # formatter_class For customization to help document input formatter-class
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.args = {'arch':[],'model':[],'data':[],'optim':[],'logs':[]}
        gargs._init_global()
        self._init_global()

    def add_argument(self,subdict,keys,**kwargs):
        if isinstance(subdict,str):
            subdict = [subdict]
        if isinstance(keys,str):
            keys = [keys]
        for k in keys:
            if k.startswith('--'):
                for d in subdict:
                    self.args[d].append(k[2:])
        self.parser.add_argument(*keys,**kwargs)

    def _init_base(self):
        pass

    # call add_argument() to add parser
    def _init_parser(self):
        # add parser
        self.add_argument('arch','--id', type=str, default='PanoNVS', help='the id to save and load model')
        self.add_argument('arch','--suffix', type=str, default=None, help='the id to save and load model')
        self.add_argument(['arch','model'],'--model', type=str, default='PanoNVS', help='the model name')
        self.add_argument(['arch','data'],'--dataset', type=str, default='PNVS', help='the dataset used')
        self.add_argument('arch','--mode', type=str, default='train', help='train | val | test')
        self.add_argument('arch','--rand', default=False, help='random')
        self.add_argument('arch','--eval', default=False, help='')
        self.add_argument('arch','--gpus', type=str, default=None, help='gpus')
        self.add_argument('arch','--world_size', type=int, default=1,help="tcp port for nccl")
        self.add_argument('arch','--port', type=int, default=9091,help="tcp port for nccl")
        self.add_argument([],'--local_rank', type=int, default=0, help="local rank for DDP")
        self.add_argument('arch','--rank', type=int, default=0,help="local rank for DDP")
        
        self.add_argument('data','--datadir', type=str, default='/home/zcq19/dataset', help='the dataset dir')
        self.add_argument('data','--imgsize', type=int, default=[256,512], nargs='+', help='the imagesize')
        self.add_argument('data','--min', type=float, default=0.3, help='')
        self.add_argument('data','--max', type=float, default=10.0, help='')
        self.add_argument('data','--batch_size', type=int, default=64, help='input batch size')
        self.add_argument('data','--threads', type=int, default=16, help='number of threads to load data')
        self.add_argument('data','--droplast', default=True, help='number of threads to load data')
        self.add_argument('data','--shuffle', action='store_false', help='shuffle the input data')

        self.add_argument('model','--padding', type=str, default='circpad', help='')

        # self.add_argument('model','--views', type=int, default=3, help='')
        # self.add_argument('model','--hypos', type=int, default=[], nargs='+', help='')
        # self.add_argument('model','--ranges', type=float, default=None, nargs='+', help='')
        # self.add_argument('model','--inter', type=str, default='lin', help='')
        # self.add_argument('model','--start', type=float, default=0.3, help='')
        # self.add_argument('model','--end', type=float, default=10.0, help='')
        # self.add_argument('model','--task', type=str, default='nvs', help='')
        # self.add_argument('model','--layers', type=int, default=3, help='')
        # self.add_argument('model','--attention', default=True, help='')
        # self.add_argument('model','--autoencoder', default=False, help='')
        
        self.add_argument('optim',['--weights','-w'], type=str, default=None, nargs='*', action=ParseKwargs)
        self.add_argument('optim','--supervise', type=str, default=None, help='optimizer')

        self.add_argument('optim','--optim', type=str, default='Adam', help='optimizer')
        self.add_argument('optim','--beta1', type=float, default=0.9, help='')
        self.add_argument('optim','--beta2', type=float, default=0.999, help='')
        self.add_argument('optim','--epochs', type=int, default=100, help='the all epochs')
        self.add_argument('optim','--lr', type=float, default=1e-4, help='learning rate')
        self.add_argument('optim','--lr_decay', type=float, default=1.0, help='learning rate decay')
        self.add_argument('optim','--weight_decay', type=float, default=0, help='learning rate decay')
        self.add_argument('optim','--lr_update', type=int, default=-1, help='learning rate update frequency')
        self.add_argument('optim','--lr_min', type=float, default=1e-6, help='min learning rate')
        self.add_argument('optim','--unused', default=False, help='find unused parameters')
        # self.add_argument('optim','--grad_clip', type=float, default=0, help='')

        self.add_argument('logs','--overview', default=False, help='')
        self.add_argument('logs','--checkpoints', type=str, default='./checkpoints', help='models are saved here')
        self.add_argument('logs','--ckpt_update', type=int, default=1, help='frequency to save model')
        self.add_argument('logs','--logs', type=str, default='./logs', help='training information are saved here')
        self.add_argument('logs',['--load_epoch', '-le'], type=int, default=-1, help='select checkpoint, default is the latest')
        self.add_argument('logs',['--reload', '-r'], action='store_true', help='resume from checkpoint')
        self.add_argument('logs','--reset_epoch', default=False, help='reset epoch')

        self.add_argument('logs','--timestamp', default='', help='reset epoch')
        self.add_argument('logs','--debug', default=False, help='save images when testing/evaluating')
        self.add_argument('logs','--savedir', type=str, default='./results', help='dir to save the results')
        self.add_argument('logs','--saveimg', default=True, help='save images when testing/evaluating')

    def paser(self,log=True):
        self._init_parser()
        self.opt = self.parser.parse_args()

        for k,v in vars(self.opt).items():
            if isinstance(v,str):
                if v.lower() == 'true':
                    setattr(self.opt,k,True)
                elif v.lower() == 'false':
                    setattr(self.opt,k,False)
                elif v.lower() == 'none':
                    setattr(self.opt,k,None)

        self.opt.imgsize = tuple(self.opt.imgsize)
        if isinstance(self.opt.hypos,int): self.opt.hypos = [self.opt.hypos]
        if isinstance(self.opt.ranges,float): self.opt.ranges = [self.opt.ranges]

        if 'PNVS' in self.opt.dataset:
            self.opt.datadir = os.path.join(self.opt.datadir,'PNVS')
        elif 'Replica' in self.opt.dataset:
            # self.opt.datadir = os.path.join(self.opt.datadir,'Replica360')
            self.opt.datadir = os.path.join(self.opt.datadir,'MatryODShkaReplica360')
        elif '3D60' in self.opt.dataset:
            self.opt.datadir = os.path.join(self.opt.datadir,'3D60')
        elif 'Matterport3D' in self.opt.dataset:
            self.opt.datadir = os.path.join(self.opt.datadir,'Matterport3D')
        elif 'Structured3D' in self.opt.dataset:
            self.opt.datadir = os.path.join(self.opt.datadir,'Structured3D')
        else:
            self.opt.datadir = os.path.join(self.opt.datadir,self.opt.dataset)
        
        if 'LOCAL_RANK' in os.environ: self.opt.local_rank = int(os.environ['LOCAL_RANK'])
        self.opt.rank = self.opt.local_rank
        if self.opt.local_rank == 0: os.makedirs(self.opt.checkpoints,exist_ok=True)
        
        if self.opt.eval: self.opt.mode = 'test'
        if self.opt.load_epoch >= 0: self.opt.reload = True

        if self.opt.threads == 0:
            self.opt.threads = self.opt.batch_size

        assert not self.opt.gpus is None or "CUDA_VISIBLE_DEVICES" in os.environ
        if not self.opt.gpus is None: os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpus
        if self.opt.gpus is None: self.opt.gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["CUDA_LAUNCH_BLOCKING"]="1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
        self.opt.world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        if self.opt.local_rank == 0:
            print("*"*80)
            print("!!! Note! Process with {} gpus, the batch size {} is divided to {} in each gpu!".format(self.opt.world_size,self.opt.batch_size,self.opt.batch_size//self.opt.world_size))
            print("*"*80)
        self.opt.batch_size = self.opt.batch_size//self.opt.world_size
        self.opt.threads = self.opt.threads//self.opt.world_size
        self.opt.droplast = self.opt.world_size > 1

        torch_seed = int(math.modf(time.time())[0]*1e8) if self.opt.rand else self.opt.local_rank
        numpy_seed = int(math.modf(time.time())[0]*1e8) if self.opt.rand else self.opt.local_rank
        random_seed = int(math.modf(time.time())[0]*1e8) if self.opt.rand else self.opt.local_rank
        python_seed = int(math.modf(time.time())[0]*1e8) if self.opt.rand else self.opt.local_rank
        
        msg = 'Local rank {}, torch seed {}, numpy seed {}, random seed {}, python seed {}'.format(
                self.opt.local_rank,torch_seed,numpy_seed,random_seed,python_seed)
        print(msg)
        
        import random
        import numpy
        import torch

        os.environ['PYTHONHASHSEED'] = str(python_seed)
        random.seed(random_seed)
        numpy.random.seed(numpy_seed)

        torch.cuda.set_device(self.opt.local_rank)
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed(torch_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()

        import torch.distributed as dist
        dist.init_process_group(backend='nccl',init_method='tcp://127.0.0.1:{}'.format(self.opt.port),rank=self.opt.local_rank,world_size=self.opt.world_size)
        if self.opt.local_rank == 0: print('Initialization success: ',dist.is_initialized())
        torch.cuda.empty_cache()

        assert self.opt.optim in ['Adam', 'SGD']
        
        self.opt.id = self.opt.id.lower()
        self.opt.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        self._set_global()

        if gargs._args['arch']['rank'] == 0:
            for subdict in gargs._args:
                print("*"*80)
                print('{}:'.format(subdict))
                for key in gargs._args[subdict]:
                    print("    {:<16}: {}".format(key,gargs._args[subdict][key]))
            print("*"*80)
        
        if gargs._args['arch']['rank'] == 0:
            os.makedirs(os.path.join(gargs._args['logs']['checkpoints'],gargs._args['arch']['id'],'config'),exist_ok=True)
            config = os.path.join(gargs._args['logs']['checkpoints'],gargs._args['arch']['id'],'config','config-{}.yaml'.format(gargs._args['logs']['timestamp']))
            gargs._args['arch']['config'] = config
            gargs._args['arch']['config_text'] = yaml.dump(gargs._args)
            print('save config to {} ...'.format(config))
            with open(config,'w') as f:
                yaml.dump(gargs._args,f)

