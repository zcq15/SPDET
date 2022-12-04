import torch
import numpy as np
import cv2

import os
import argparse

from model.spdet import SPDET
from utils.utils import tensor2img

class Options(object):

    def __init__(self):
        # create ArgumentParser() obj
        # formatter_class For customization to help document input formatter-class
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self._init_parser()
        self.conf = self.parser.parse_args()

    # call add_argument() to add parser
    def _init_parser(self):
        # add parser
        self.parser.add_argument('--checkpoints', type=str, default='./checkpoints/spdet-3d60.pt')
        self.parser.add_argument('--image', type=str, default='./images/3d60.png')
        self.parser.add_argument('--savedir', type=str, default='./results')


if __name__ == '__main__':
    CONFIG = Options()
    model = SPDET().cuda()
    print('load checkpoints from '+CONFIG.conf.checkpoints)
    state_dict = torch.load(CONFIG.conf.checkpoints,map_location='cpu')
    model.load_state_dict(state_dict['model'])

    print('load image from '+CONFIG.conf.image)
    img = cv2.imread(CONFIG.conf.image,cv2.IMREAD_COLOR)
    img = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),[512,256],cv2.INTER_CUBIC)
    img = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()/255.0
    
    with torch.no_grad():
        depth_tensor = model(img)
        depth_turbo = tensor2img(depth_tensor[0],imtype='turbo')
        depth_gray = tensor2img(depth_tensor[0],imtype='depth')
        img = tensor2img(img[0],imtype='color')
    
    os.makedirs(CONFIG.conf.savedir,exist_ok=True)
    print('save results to '+CONFIG.conf.savedir)
    fn = os.path.basename(CONFIG.conf.image).split('.')[0]
    cv2.imwrite(os.path.join(CONFIG.conf.savedir,fn+'_depth.png'),depth_turbo)
    cv2.imwrite(os.path.join(CONFIG.conf.savedir,fn+'_depth_gray.png'),depth_gray)
    cv2.imwrite(os.path.join(CONFIG.conf.savedir,fn+'_input.png'),img)
