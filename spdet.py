from tqdm import tqdm
from models.options import Options
Options().paser()
from models.gargs import _args
import torch
from torch.utils.data import DataLoader

from models.network import Network
from datasets.pnvsdepth import PNVSDepth,EasyPNVSDepth,HardPNVSDepth
from datasets.threed60 import ThreeD60,ThreeD60M3D,ThreeD60S2D3D
# from datasets.panosuncg import PanoSUNCG

from models.metrics.metrics import Photometric,Perceptual,UVEdge,UVSmooth,BerHu,Smooth
from models.metrics.metrics_eval_perspective import L1,SqRel,RMSE,Log10RMSE,AbsRel,Delta
from models.metrics.metrics_eval_spherical import SPL1,SPAbsRel,SPSqRel,SPRMSE,SPLogERMSE,SPDelta

import os
from collections import OrderedDict
import glob

def run(funcs,weights,warmup,metrics):
    model = Network(funcs)
    dataloader_train,dataloader_test,sampler_train,sampler_test = run_init()
    test = run_exec(model,
            dataloader_train,dataloader_test,
            sampler_train,sampler_test,
            weights,warmup,metrics)
    return test

def run_init():

    dataset_dict = {
        'PNVSDepth':PNVSDepth,
        'EasyPNVSDepth':EasyPNVSDepth,
        'HardPNVSDepth':HardPNVSDepth,
        '3D60':ThreeD60,
        '3D60M3D':ThreeD60M3D,
        '3D60S2D3D':ThreeD60S2D3D,
        # 'PanoSUNCG':PanoSUNCG
    }

    dataset = dataset_dict[_args['arch']['dataset']]

    dataset_train = dataset(_args['data']['datadir'],area='train',imgsize=_args['data']['imgsize'],local_rank=_args['arch']['rank'])
    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train,_args['data']['batch_size'],num_workers=_args['data']['threads'],\
        sampler=sampler_train,pin_memory=True,drop_last=_args['data']['droplast'])

    dataset_test = dataset(_args['data']['datadir'],area='test',imgsize=_args['data']['imgsize'],local_rank=_args['arch']['rank'])
    sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test,_args['data']['batch_size'],num_workers=_args['data']['threads'],sampler=sampler_test,pin_memory=True,drop_last=_args['data']['droplast'])

    return dataloader_train,dataloader_test,sampler_train,sampler_test

def run_exec(model,
             dataloader_train,dataloader_test,
             sampler_train,sampler_test,
             weights,warmup,metrics):

    volumes_test = OrderedDict({'Size':0})
    for key in metrics:
        volumes_test[key] = 0
    if _args['arch']['mode'] == 'train':
        start = model.epoch
        end = _args['optim']['epochs']
        iters = range(start,end)
        for epoch in iters:

            _args['arch']['mode'] = 'train'
            sampler_train.set_epoch(model.epoch)
            model.train(dataloader_train,weights,warmup)

            _args['arch']['mode'] = 'test'
            sampler_test.set_epoch(model.epoch)
            model.test(dataloader_test,metrics)
            _args['arch']['mode'] = 'train'

    else:
        sampler_test.set_epoch(model.epoch)
        model.test(dataloader_test,metrics)

    del model,dataloader_train,dataloader_test
    torch.cuda.empty_cache()
    return volumes_test

if __name__ == '__main__':

    if _args['optim']['supervise'] == 'scatter':

        weights = _args['optim']['weights'] or {'P':1.0,'V':0.01,'E':0.01,'S':0.01}
        warmup = {}
        metrics =['WAbs','WSq','WMAE','WRMSE','WLogERMSE','WD1','WD2','WD3']

        funcs = OrderedDict({
            'P':Photometric(branch='tar_rgb_scatter',target='target',valid='tar_valid_scatter'),
            'V':Perceptual(branch='tar_rgb_scatter',target='target',valid='tar_valid_scatter'),
            'E':UVEdge(branch='tar_uv_scatter',target='source'),
            'S':UVSmooth(branch='tar_uv_scatter',target='source'),

            'MAE':L1(max=_args['data']['max']),'RMSE':RMSE(max=_args['data']['max']),'Log10RMSE':Log10RMSE(max=_args['data']['max']),
            'Abs':AbsRel(max=_args['data']['max']),
            'D1':Delta(th=1.25,max=_args['data']['max']),'D2':Delta(th=1.25**2,max=_args['data']['max']),'D3':Delta(th=1.25**3,max=_args['data']['max']),

            'WMAE':SPL1(max=_args['data']['max']),'WRMSE':SPRMSE(max=_args['data']['max']),'WLogERMSE':SPLogERMSE(max=_args['data']['max']),
            'WAbs':SPAbsRel(max=_args['data']['max']),'WSq':SPSqRel(max=_args['data']['max']),
            'WD1':SPDelta(th=1.25,max=_args['data']['max']),'WD2':SPDelta(th=1.25**2,max=_args['data']['max']),'WD3':SPDelta(th=1.25**3,max=_args['data']['max'])

        })

    elif _args['optim']['supervise'] == 'inter':

        weights = _args['optim']['weights'] or {'P':1.0,'V':0.01,'S':0.01}
        warmup = {}
        metrics =['WAbs','WSq','WMAE','WRMSE','WLogERMSE','WD1','WD2','WD3']

        funcs = OrderedDict({
            'P':Photometric(branch='src_rgb_inter',target='source'),
            'V':Perceptual(branch='src_rgb_inter',target='source'),
            'S':Smooth(branch='src_depth',target='source'),

            'MAE':L1(max=_args['data']['max']),'RMSE':RMSE(max=_args['data']['max']),'Log10RMSE':Log10RMSE(max=_args['data']['max']),
            'Abs':AbsRel(max=_args['data']['max']),
            'D1':Delta(th=1.25,max=_args['data']['max']),'D2':Delta(th=1.25**2,max=_args['data']['max']),'D3':Delta(th=1.25**3,max=_args['data']['max']),

            'WMAE':SPL1(max=_args['data']['max']),'WRMSE':SPRMSE(max=_args['data']['max']),'WLogERMSE':SPLogERMSE(max=_args['data']['max']),
            'WAbs':SPAbsRel(max=_args['data']['max']),'WSq':SPSqRel(max=_args['data']['max']),
            'WD1':SPDelta(th=1.25,max=_args['data']['max']),'WD2':SPDelta(th=1.25**2,max=_args['data']['max']),'WD3':SPDelta(th=1.25**3,max=_args['data']['max'])

        })

    else:

        weights = _args['optim']['weights'] or {'B':1.0}
        warmup = {}
        metrics =['Abs','SqRel','MAE','RMSE','Log10RMSE','D1','D2','D3']

        funcs = OrderedDict({
            'B':BerHu(max=_args['data']['max']),'L1':L1(max=_args['data']['max']),

            'MAE':L1(max=_args['data']['max']),'RMSE':RMSE(max=_args['data']['max']),'Log10RMSE':Log10RMSE(max=_args['data']['max']),'Abs':AbsRel(max=_args['data']['max']), 'SqRel':SqRel(max=_args['data']['max']),
            'D1':Delta(th=1.25,max=_args['data']['max']),'D2':Delta(th=1.25**2,max=_args['data']['max']),'D3':Delta(th=1.25**3,max=_args['data']['max'])
        })

    run(funcs,weights,warmup,metrics)

