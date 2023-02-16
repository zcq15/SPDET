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
        self.add_argument(['arch','data'],'--testset', type=str, default=None, help='the testset used')
        self.add_argument('arch','--mode', type=str, default='train', help='train | val | test')
        self.add_argument('arch','--rand', default=False, help='random')
        self.add_argument('arch','--eval', default=False, help='')
        self.add_argument('arch','--gpus', type=str, default=None, help='gpus')
        self.add_argument('arch','--world_size', type=int, default=1,help="tcp port for nccl")
        self.add_argument('arch','--port', type=int, default=9091,help="tcp port for nccl")
        self.add_argument([],'--local_rank', type=int, default=0, help="local rank for DDP")
        self.add_argument('arch','--rank', type=int, default=0,help="local rank for DDP")

        self.add_argument('data','--datadir', type=str, default='/home/zcq19/dataset', help='the dataset dir')
        self.add_argument('data','--testdir', type=str, default='/home/zcq19/dataset', help='the dataset dir')
        self.add_argument('data','--imgsize', type=int, default=None, nargs='+', help='the imagesize')
        self.add_argument('data','--trainsize', type=int,  default=None, nargs='+', help='the imagesize')
        self.add_argument('data','--testsize', type=int, default=None, nargs='+', help='the imagesize')
        self.add_argument('data','--min', type=float, default=0.1, help='')
        self.add_argument('data','--max', type=float, default=10.0, help='')
        self.add_argument('data','--batch_size', type=int, default=64, help='input batch size')
        self.add_argument('data','--threads', type=int, default=16, help='number of threads to load data')
        self.add_argument('data','--droplast', type=bool,default=True, help='number of threads to load data')
        self.add_argument('data','--shuffle', action='store_false', help='shuffle the input data')

        self.add_argument('model','--padding', type=str, default='circpad', help='')

        self.add_argument('optim',['--weights','-w'], type=str, default=None, nargs='*', action=ParseKwargs)
        self.add_argument('optim','--weighted', default=True, help='')
        self.add_argument('optim','--supervise', type=str, default='scatter', help='')
        self.add_argument('optim','--align', default=False, help='')
        self.add_argument('optim','--upscale', type=int, default=2, help='')
        self.add_argument('optim','--downscale', type=int, default=2, help='')
        self.add_argument('optim','--wind', type=int, default=7, help='')
        self.add_argument('optim','--std', type=float, default=1.5, help='')
        self.add_argument('optim','--dist', type=str, default='l1', help='')
        self.add_argument('optim','--view', type=str, default='cube', help='')
        self.add_argument('optim','--filter', default=True, help='')
        self.add_argument('optim','--pmth', type=float, default=0.5, help='')
        self.add_argument('optim','--pmwarmup', type=int, default=0, help='')

        self.add_argument('optim','--optim', type=str, default='Adam', help='optimizer')
        self.add_argument('optim','--load_optim', default=False, help='')
        self.add_argument('optim','--beta1', type=float, default=0.9, help='')
        self.add_argument('optim','--beta2', type=float, default=0.999, help='')
        self.add_argument('optim','--epochs', type=int, default=30, help='the all epochs')
        self.add_argument('optim','--lr', type=float, default=1e-4, help='learning rate')
        self.add_argument('optim','--lr_decay', type=float, default=0.1, help='learning rate decay')
        self.add_argument('optim','--weight_decay', type=float, default=0, help='learning rate decay')
        self.add_argument('optim','--lr_update', type=int, default=-1, help='learning rate update frequency')
        self.add_argument('optim','--lr_patience', type=int, default=10, help='learning rate update frequency')
        self.add_argument('optim','--lr_min', type=float, default=1e-5, help='min learning rate')
        self.add_argument('optim','--unused', default=False, help='find unused parameters')

        self.add_argument('logs','--overview', default=False, help='')
        self.add_argument('logs','--checkpoints', type=str, default='./checkpoints', help='models are saved here')
        self.add_argument('logs','--ckpt_update', type=int, default=1, help='frequency to save model')
        self.add_argument('logs','--logs', type=str, default='./logs', help='training information are saved here')
        self.add_argument('logs',['--load_epoch', '-le'], type=int, default=-1, help='select checkpoint, default is the latest')
        self.add_argument('logs',['--reload', '-r'], action='store_true', help='resume from checkpoint')
        self.add_argument('logs',['--reset_epoch', '-re'], default=False, help='reset epoch')

        self.add_argument('logs','--timestamp', default='', help='reset epoch')
        self.add_argument('logs','--debug', default=False, help='save images when testing/evaluating')
        self.add_argument('logs','--savedir', type=str, default='./results', help='dir to save the results')
        self.add_argument('logs','--saveimg', default=True, help='save images when testing/evaluating')


    def get_imgsize(self,dataset):
        if 'PNVS' in dataset:
            return (256,512)
        elif '3D60' in dataset:
            return (256,512)
        elif 'Stanford2D3DS' in dataset:
            return (512,1024)
        elif 'Matterport3D' in dataset:
            return (512,1024)
        elif 'PanoSUNCG' in dataset:
            return (256,512)
        elif '360SD' in dataset:
            return (256,512)
        elif 'Structured3D' in dataset:
            return (256,512)
        elif 'Joint360Depth' in dataset:
            return (256,512)
        elif '360VO' in dataset:
            return (256,512)
        elif '360Lightfield' in dataset:
            return (256,512)
        else:
            return (256,512)
    
    def get_datadir(self,dataset,datadir):
        if 'PNVS' in dataset:
            return os.path.join(datadir,'PNVS')
        elif '3D60' in dataset:
            return os.path.join(datadir,'3D60')
        elif 'Stanford2D3DS' in dataset:
            return os.path.join(datadir,'Stanford2D3DS')
        elif 'Matterport3D' in dataset:
            return os.path.join(datadir,'Matterport3D')
        elif 'PanoSUNCG' in dataset:
            return os.path.join(datadir,'PanoSUNCG')
        elif '360SD' in dataset:
            return os.path.join(datadir,'360SD')
        elif 'Structured3D' in dataset:
            return os.path.join(datadir,'Structured3D')
        elif 'Joint360Depth' in dataset:
            return os.path.join(datadir,'Joint360Depth')
        elif '360VO' in dataset:
            return os.path.join(datadir,'360VO')
        elif '360Lightfield' in dataset:
            return os.path.join(datadir,'360Lightfield')
        else:
            return os.path.join(datadir,dataset)

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

        if self.opt.imgsize is None: self.opt.imgsize = self.get_imgsize(self.opt.dataset)
        if self.opt.testset is None: self.opt.testset = self.opt.dataset
        if self.opt.testdir is None: self.opt.testdir = self.opt.datadir
        self.opt.datadir = self.get_datadir(self.opt.dataset,self.opt.datadir)
        self.opt.testdir = self.get_datadir(self.opt.testset,self.opt.testdir)

        if 'LOCAL_RANK' in os.environ: self.opt.local_rank = int(os.environ['LOCAL_RANK'])
        self.opt.rank = self.opt.local_rank
        if self.opt.local_rank == 0: os.makedirs(self.opt.checkpoints,exist_ok=True)
        
        if self.opt.eval: self.opt.mode = 'test'
        if self.opt.load_epoch >= 0: self.opt.reload = True

        if self.opt.threads == 0:
            self.opt.threads = self.opt.batch_size

        # assert not self.opt.gpus is None or "CUDA_VISIBLE_DEVICES" in os.environ
        # if not self.opt.gpus is None: os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpus
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

