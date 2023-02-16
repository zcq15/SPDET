import torch
import torch.nn as nn

class L1(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max
        self.weights = None

    def forward(self,_output,_input):
        if not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']
        else:
            mask = torch.ones_like(_input['src_depth'],dtype=torch.bool,device=_input['src_depth'].device)

        B,C,H,W = _input['src_rgb'].shape
        loss = 0
        for b in range(B):
            loss += torch.sum((pred[b]-target[b]).abs()*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum()
        return loss/B

class MSE(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max

    def forward(self,_output,_input):
        if not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']
        else:
            mask = torch.ones_like(_input['src_depth'],dtype=torch.bool,device=_input['src_depth'].device)

        B,C,H,W = _input['src_rgb'].shape
        loss = 0
        for b in range(B):
            loss += torch.sum((pred[b]-target[b]).pow(2)*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum()
        return loss/B

class AbsRel(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max

    def forward(self,_output,_input):
        if not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']*(target>0)
        else:
            mask = target>0

        B,C,H,W = _input['src_rgb'].shape
        loss = 0
        for b in range(B):
            target[b][~mask[b]] = 1.0
            loss += torch.sum((pred[b]-target[b]).abs()/target[b]*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum()
        return loss/B

class SqRel(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max

    def forward(self,_output,_input):
        if not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']*(target>0)
        else:
            mask = target>0

        B,C,H,W = _input['src_rgb'].shape
        loss = 0
        for b in range(B):
            target[b][~mask[b]] = 1.0
            loss += torch.sum((pred[b]-target[b]).pow(2)/target[b]*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum()
        return loss/B

class RMSE(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max

    def forward(self,_output,_input):
        if not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']
        else:
            mask = torch.ones_like(_input['src_depth'],dtype=torch.bool,device=_input['src_depth'].device)

        B,C,H,W = _input['src_rgb'].shape
        loss = 0
        for b in range(B):
            loss += torch.sqrt(torch.sum((pred[b]-target[b]).pow(2)*mask[b].type(torch.float32))/mask[b].type(torch.float32).sum())
        return loss/B

class Log10RMSE(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0.0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max

    def forward(self,_output,_input):
        if not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)+1e-5
        target = _input[self.target].clip(self.min,self.max)+1e-5
        if 'src_valid' in _input:
            mask = _input['src_valid']*(pred>0)*(target>0)
        else:
            mask = (pred>0)*(target>0)

        B,C,H,W = _input['src_rgb'].shape
        loss = 0
        for b in range(B):
           delta = (torch.log10(pred[b]) - torch.log10(target[b])).pow(2)
           delta[~mask[b]] = 0
           loss += torch.sqrt( torch.sum(delta*mask[b].type(torch.float32)) / mask[b].type(torch.float32).sum() )
        return loss/B

class LogERMSE(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',min=0.0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.min = min
        self.max = max

    def forward(self,_output,_input):
        if not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)
        target = _input[self.target].clip(self.min,self.max)
        if 'src_valid' in _input:
            mask = _input['src_valid']*(pred>0)*(target>0)
        else:
            mask = (pred>0)*(target>0)

        B,C,H,W = _input['src_rgb'].shape
        loss = 0
        for b in range(B):
           delta = (torch.log(pred[b]) - torch.log(target[b])).pow(2)
           delta[~mask[b]] = 0
           loss += torch.sqrt( torch.sum(delta*mask[b].type(torch.float32)) / mask[b].type(torch.float32).sum() )
        return loss/B


class Delta(nn.Module):
    def __init__(self,branch='src_depth',target='src_depth',th=1.25,min=0,max=10.0):
        super().__init__()
        self.branch = branch
        self.target = target
        self.th = th
        self.min = min
        self.max = max
    def forward(self,_output,_input):
        if not self.branch in _output:
            return torch.tensor(0,dtype=torch.float32).cuda()
        pred = _output[self.branch].clip(self.min,self.max)+1e-5
        target = _input[self.target].clip(self.min,self.max)+1e-5
        if 'src_valid' in _input:
            mask = _input['src_valid']*(pred>0)*(target>0)
        else:
            mask = (pred>0)*(target>0)

        B,C,H,W = _input['src_rgb'].shape
        loss = 0
        for b in range(B):
            thresh = torch.max(target[b]/pred[b], pred[b]/target[b])
            thresh[~mask[b]] = self.th + 1.0
            mask_sum = torch.sum(mask[b].type(torch.float32))
            loss += (thresh<self.th).type(torch.float32).sum()/mask_sum
        return loss/B
