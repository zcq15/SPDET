import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from packaging.version import Version

import cv2
import numpy as np
import math

class Equirec2Cube(torch.nn.Module):
    def __init__(self, equ_h, equ_w, out_dim, FOV=90, RADIUS=128, CUDA=True):
        super().__init__()
        batch_size = 1
        RADIUS = int(0.5*out_dim)
        R_lst = []
        theta_lst = np.array([-90, 0, 90, 180], np.float32) / 180 * np.pi
        phi_lst = np.array([90, -90], np.float32) / 180 * np.pi
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.CUDA = CUDA
        for theta in theta_lst:
            angle_axis = theta * np.array([0, 1, 0], np.float32)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)

        for phi in phi_lst:
            angle_axis = phi * np.array([1, 0, 0], np.float32)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)
        
        R_lst = [Variable(torch.FloatTensor(x)) for x in R_lst]
        
        self.out_dim = out_dim
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0
        c_x = (out_dim - 1) / 2.0
        c_y = (out_dim - 1) / 2.0
        
        wangle = (180 - FOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(FOV / 2.0)) / np.sin(np.radians(wangle))

        f = RADIUS / w_len * out_dim
        cx = c_x
        cy = c_y
        self.intrisic = {
                    'f': float(f),
                    'cx': float(cx),
                    'cy': float(cy)
                }
        #self.R_lst = R_lst

        interval = w_len / (out_dim - 1) 
        
        z_map = np.zeros([out_dim, out_dim], np.float32) + RADIUS
        x_map = np.tile((np.arange(out_dim) - c_x) * interval, [out_dim, 1])
        y_map = np.tile((np.arange(out_dim) - c_y) * interval, [out_dim, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([out_dim, out_dim, 3], np.float32)
        xyz[:, :, 0] = (RADIUS / D) * x_map[:, :]
        xyz[:, :, 1] = (RADIUS / D) * y_map[:, :]
        xyz[:, :, 2] = (RADIUS / D) * z_map[:, :]

        xyz = Variable(torch.FloatTensor(xyz))
        
        reshape_xyz = xyz.view(out_dim * out_dim, 3).transpose(0, 1)
        self.batch_size = batch_size # NOTE: Might give an error when batch_size smaller than real batch_size of the batch input
        self.loc = []
        self.grid = []
        for i, R in enumerate(R_lst):
            result = torch.matmul(R, reshape_xyz).transpose(0, 1)
            tmp_xyz = result.contiguous().view(1, out_dim, out_dim, 3)
            self.grid.append(tmp_xyz)
            lon = torch.atan2(result[:, 0] , result[:, 2]).view(1, out_dim, out_dim, 1) / np.pi
            lat = torch.asin(result[:, 1] / RADIUS).view(1, out_dim, out_dim, 1) / (np.pi / 2)

            self.loc.append(torch.cat([lon.repeat(batch_size, 1, 1, 1), lat.repeat(batch_size, 1, 1, 1)], dim=3))

        new_lst = [3, 5, 1, 0, 2, 4]
        self.R_lst = [R_lst[x] for x in new_lst]
        #self.grid_lst = [self.grid [x] for x in new_lst]
        self.grid_lst = []
        for iii in new_lst:
            grid = self.grid[iii].clone()
            scale = self.intrisic['f'] / grid[:, :, :, 2:3]
            grid *= scale
            self.grid_lst.append(grid)

    def _ToCube(self, batch, mode):
        batch_size = batch.size()[0]
        #if batch_size != self.batch_size:
        #    print('Batch error! Expect to have {} but got {}'.format(self.batch_size, batch_size))

        #lst = ['left', 'front', 'right', 'back', 'up', 'down']
        #lst = ['back', 'down', 'front', 'left', 'right', 'up']
        new_lst = [3, 5, 1, 0, 2, 4]
        out = []
        for i in new_lst:
            coor = self.loc[i].cuda() if self.CUDA else self.loc[i]
            result = []
            for ii in range(batch_size):
                tmp = F.grid_sample(batch[ii:ii+1], coor, mode=mode,align_corners=True)
                result.append(tmp)
            result = torch.cat(result, dim=0)
            out.append(result)
        return out

    def GetGrid(self):
        #lst = ['left', 'front', 'right', 'back', 'up', 'down']
        new_lst = [3, 5, 1, 0, 2, 4]
        out = [self.grid[x] for x in new_lst]
        out = torch.cat(out, dim=0)
        return out


    def ToCubeNumpy(self, batch):
        out = self._ToCube(batch)
        result = [x.data.cpu().numpy() for x in out]
        return result

    def ToCubeTensor(self, batch, mode='bilinear'):
        assert mode in ['bilinear', 'nearest']
        batch_size = batch.size()[0]
        cube = self._ToCube(batch, mode=mode)
        out_batch = None
        for batch_idx in range(batch_size):
            for cube_idx in range(6):
                patch = torch.unsqueeze(cube[cube_idx][batch_idx, :, :, :], 0)
                if out_batch is None:
                    out_batch = patch
                else:
                    out_batch = torch.cat([out_batch, patch], dim=0)
        #print out_batch
        #exit()
        return out_batch
    
    def forward(self,x):
        return self.ToCubeTensor(x)


class Cube2Equirec(torch.nn.Module):
    def __init__(self, cube_size, output_h, output_w, cube_fov=90, CUDA=True):
        super().__init__()
        self.batch_size = 1 # NOTE: not in use at all
        self.cube_h = cube_size
        self.cube_w = cube_size
        self.output_h = output_h
        self.output_w = output_w
        self.fov = cube_fov
        self.fov_rad = self.fov * np.pi / 180
        self.CUDA = CUDA

        # Compute the parameters for projection
        assert self.cube_w == self.cube_h
        self.radius = int(0.5 * cube_size)

        # Map equirectangular pixel to longitude and latitude
        # NOTE: Make end a full length since arange have a right open bound [a, b)
        theta_start = math.pi - (math.pi / output_w)
        theta_end = -math.pi
        theta_step = 2 * math.pi / output_w
        theta_range = torch.arange(theta_start, theta_end, -theta_step)

        phi_start = 0.5 * math.pi - (0.5 * math.pi / output_h)
        phi_end = -0.5 * math.pi
        phi_step = math.pi / output_h
        phi_range = torch.arange(phi_start, phi_end, -phi_step)

        # Stack to get the longitude latitude map
        self.theta_map = theta_range.unsqueeze(0).repeat(output_h, 1)
        self.phi_map = phi_range.unsqueeze(-1).repeat(1, output_w)
        self.lonlat_map = torch.stack([self.theta_map, self.phi_map], dim=-1)

        # Get mapping relation (h, w, face)
        # [back, down, front, left, right, up] => [0, 1, 2, 3, 4, 5]
        # self.orientation_mask = self.get_orientation_mask()

        # Project each face to 3D cube and convert to pixel coordinates
        self.grid, self.orientation_mask = self.get_grid2()

        if self.CUDA:
            self.grid = self.grid.cuda()
            self.orientation_mask = self.orientation_mask.cuda()

    # Compute the orientation mask for the lonlat map
    def get_orientation_mask(self):
        mask_back_lon = (self.lonlat_map[:, :, 0] > np.pi - 0.5 * self.fov_rad) + \
                        (self.lonlat_map[:, :, 0] < - np.pi + 0.5 * self.fov_rad)
        mask_back_lat = (self.lonlat_map[:, :, 1] < 0.5 * self.fov_rad) * \
                        (self.lonlat_map[:, :, 1] > - 0.5 * self.fov_rad)
        mask_back = mask_back_lon * mask_back_lat

        mask_down_lat = (self.lonlat_map[:, :, 1] <= - 0.5 * self.fov_rad)
        mask_down = mask_down_lat

        mask_front_lon = (self.lonlat_map[:, :, 0] < 0.5 * self.fov_rad) * \
                         (self.lonlat_map[:, :, 0] > - 0.5 * self.fov_rad)
        mask_front_lat = (self.lonlat_map[:, :, 1] < 0.5 * self.fov_rad) * \
                         (self.lonlat_map[:, :, 1] > - 0.5 * self.fov_rad)
        mask_front = mask_front_lon * mask_front_lat

        mask_left_lon = (self.lonlat_map[:, :, 0] < np.pi - 0.5 * self.fov_rad) * \
                        (self.lonlat_map[:, :, 0] > 0.5 * self.fov_rad)
        mask_left_lat = (self.lonlat_map[:, :, 1] < 0.5 * self.fov_rad) * \
                        (self.lonlat_map[:, :, 1] > - 0.5 * self.fov_rad)
        mask_left = mask_left_lon * mask_left_lat

        mask_right_lon = (self.lonlat_map[:, :, 0] < - 0.5 * self.fov_rad) * \
                         (self.lonlat_map[:, :, 0] > - np.pi + 0.5 * self.fov_rad)
        mask_right_lat = (self.lonlat_map[:, :, 1] < 0.5 * self.fov_rad) * \
                         (self.lonlat_map[:, :, 1] > - 0.5 * self.fov_rad)
        mask_right = mask_right_lon * mask_right_lat

        # mask_up_lat = (self.lonlat_map[:, :, 1] >= 0.5 * self.fov_rad)
        mask_up = torch.ones([self.output_h, self.output_w])
        mask_up = mask_up - (self.lonlat_map[:, :, 1] < 0).float() - \
                  (mask_front.float() + mask_right.float() + mask_left.float() + mask_back.float())
        mask_up = (mask_up == 1)

        # Face map contains numbers correspond to that face
        orientation_mask = mask_back * 0 + mask_down * 1 + mask_front * 2 + mask_left * 3 + mask_right * 4 + mask_up * 5

        return orientation_mask

    def get_grid2(self):
        # Get the point of equirectangular on 3D ball
        x_3d = (self.radius * torch.cos(self.phi_map) * torch.sin(self.theta_map)).view(self.output_h, self.output_w, 1)
        y_3d = (self.radius * torch.sin(self.phi_map)).view(self.output_h, self.output_w, 1)
        z_3d = (self.radius * torch.cos(self.phi_map) * torch.cos(self.theta_map)).view(self.output_h, self.output_w, 1)

        self.grid_ball = torch.cat([x_3d, y_3d, z_3d], 2).view(self.output_h, self.output_w, 3)

        # Compute the down grid
        radius_ratio_down = torch.abs(y_3d / self.radius)
        grid_down_raw = self.grid_ball / radius_ratio_down.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_down_w = (-grid_down_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_down_h = (-grid_down_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_down = torch.cat([grid_down_w, grid_down_h], 2).unsqueeze(0)
        mask_down = (((grid_down_w <= 1) * (grid_down_w >= -1)) * ((grid_down_h <= 1) * (grid_down_h >= -1)) *
                    (grid_down_raw[:, :, 1] == -self.radius).unsqueeze(2)).float()

        # Compute the up grid
        radius_ratio_up = torch.abs(y_3d / self.radius)
        grid_up_raw = self.grid_ball / radius_ratio_up.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_up_w = (-grid_up_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_up_h = (grid_up_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_up = torch.cat([grid_up_w, grid_up_h], 2).unsqueeze(0)
        mask_up = (((grid_up_w <= 1) * (grid_up_w >= -1)) * ((grid_up_h <= 1) * (grid_up_h >= -1)) *
                  (grid_up_raw[:, :, 1] == self.radius).unsqueeze(2)).float()

        # Compute the front grid
        radius_ratio_front = torch.abs(z_3d / self.radius)
        grid_front_raw = self.grid_ball / radius_ratio_front.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_front_w = (-grid_front_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_front_h = (-grid_front_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_front = torch.cat([grid_front_w, grid_front_h], 2).unsqueeze(0)
        mask_front = (((grid_front_w <= 1) * (grid_front_w >= -1)) * ((grid_front_h <= 1) * (grid_front_h >= -1)) *
                  (torch.round(grid_front_raw[:, :, 2]) == self.radius).unsqueeze(2)).float()

        # Compute the back grid
        radius_ratio_back = torch.abs(z_3d / self.radius)
        grid_back_raw = self.grid_ball / radius_ratio_back.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_back_w = (grid_back_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_back_h = (-grid_back_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_back = torch.cat([grid_back_w, grid_back_h], 2).unsqueeze(0)
        mask_back = (((grid_back_w <= 1) * (grid_back_w >= -1)) * ((grid_back_h <= 1) * (grid_back_h >= -1)) *
                  (torch.round(grid_back_raw[:, :, 2]) == -self.radius).unsqueeze(2)).float()


        # Compute the right grid
        radius_ratio_right = torch.abs(x_3d / self.radius)
        grid_right_raw = self.grid_ball / radius_ratio_right.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_right_w = (-grid_right_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_right_h = (-grid_right_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_right = torch.cat([grid_right_w, grid_right_h], 2).unsqueeze(0)
        mask_right = (((grid_right_w <= 1) * (grid_right_w >= -1)) * ((grid_right_h <= 1) * (grid_right_h >= -1)) *
                  (torch.round(grid_right_raw[:, :, 0]) == -self.radius).unsqueeze(2)).float()

        # Compute the left grid
        radius_ratio_left = torch.abs(x_3d / self.radius)
        grid_left_raw = self.grid_ball / radius_ratio_left.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_left_w = (grid_left_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_left_h = (-grid_left_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_left = torch.cat([grid_left_w, grid_left_h], 2).unsqueeze(0)
        mask_left = (((grid_left_w <= 1) * (grid_left_w >= -1)) * ((grid_left_h <= 1) * (grid_left_h >= -1)) *
                  (torch.round(grid_left_raw[:, :, 0]) == self.radius).unsqueeze(2)).float()

        # Face map contains numbers correspond to that face
        orientation_mask = mask_back * 0 + mask_down * 1 + mask_front * 2 + mask_left * 3 + mask_right * 4 + mask_up * 5

        return torch.cat([grid_back, grid_down, grid_front, grid_left, grid_right, grid_up], 0), orientation_mask

    # Convert cubic images to equirectangular
    def _ToEquirec(self, batch, mode):
        batch_size, ch, H, W = batch.shape
        if batch_size != 6:
            raise ValueError("Batch size mismatch!!")

        if self.CUDA:
            output = Variable(torch.zeros(1, ch, self.output_h, self.output_w), requires_grad=False).cuda()
        else:
            output = Variable(torch.zeros(1, ch, self.output_h, self.output_w), requires_grad=False)

        for ori in range(6):
            grid = self.grid[ori, :, :, :].unsqueeze(0) # 1, self.output_h, self.output_w, 2
            mask = (self.orientation_mask == ori).unsqueeze(0) # 1, self.output_h, self.output_w, 1

            if self.CUDA:
                masked_grid = Variable(grid * mask.float().expand(-1, -1, -1, 2)).cuda() # 1, self.output_h, self.output_w, 2
            else:
                masked_grid = Variable(grid * mask.float().expand(-1, -1, -1, 2))

            source_image = batch[ori].unsqueeze(0) # 1, ch, H, W

            sampled_image = torch.nn.functional.grid_sample(source_image, masked_gridmode=mode, align_corners=True) # 1, ch, self.output_h, self.output_w

            if self.CUDA:
                sampled_image_masked = sampled_image * \
                                    Variable(mask.float().view(1, 1, self.output_h, self.output_w).expand(1, ch, -1, -1)).cuda()
            else:
                sampled_image_masked = sampled_image * \
                                       Variable(mask.float().view(1, 1, self.output_h, self.output_w).expand(1, ch, -1, -1))
            output = output + sampled_image_masked # 1, ch, self.output_h, self.output_w

        return output

    # Convert input cubic tensor to output equirectangular image
    def ToEquirecTensor(self, batch, mode='bilinear'):
        # Check whether batch size is 6x
        assert mode in ['nearest', 'bilinear']
        batch_size = batch.size()[0]
        if batch_size % 6 != 0:
            raise ValueError("Batch size should be 6x")

        processed = []
        for idx in range(int(batch_size / 6)):
            target = batch[idx * 6 : (idx + 1) * 6, :, :, :]
            target_processed = self._ToEquirec(target, mode)
            processed.append(target_processed)

        output = torch.cat(processed, 0)
        return output
    
    def forward(self,x):
        return self.ToEquirecTensor(x)

class Depth2Radius(nn.Module):
    def __init__(self,w):
        super().__init__()
        f = w/2
        if Version(torch.__version__) >= Version('1.10.0'):
            y,x = torch.meshgrid(torch.arange(w), torch.arange(w), indexing='ij')
        else:
            y,x = torch.meshgrid(torch.arange(w), torch.arange(w))
        x = x.cuda()-(w-1)/2
        y = y.cuda()-(w-1)/2
        self.rate = (y.pow(2)+x.pow(2)+f**2).sqrt()/f
        self.rate = self.rate.unsqueeze(0).unsqueeze(0)

    def forward(self,x): 
        return x*self.rate

class Radius2Depth(nn.Module):
    def __init__(self,w):
        super().__init__()
        f = w/2
        if Version(torch.__version__) >= Version('1.10.0'):
            y,x = torch.meshgrid(torch.arange(w), torch.arange(w), indexing='ij')
        else:
            y,x = torch.meshgrid(torch.arange(w), torch.arange(w))
        x = x.cuda()-(w-1)/2
        y = y.cuda()-(w-1)/2
        self.rate = f/(y.pow(2)+x.pow(2)+f**2).sqrt()
        self.rate = self.rate.unsqueeze(0).unsqueeze(0)

    def forward(self,x): 
        return x*self.rate

class CETransform(nn.Module):
    def __init__(self, h=None, w=None, d=None):
        super().__init__()
        self.reinit(h,w,d)

    def reinit(self, h=None, w=None, d=None):
        if h is None or w is None or d is None:
            self.h = None
            self.w = None
            self.d = None
            self.e2c = None
            self.c2e = None
            self.r2d = None
            self.d2r = None
        else:
            self.h = h
            self.w = w
            self.d = d
            self.e2c = Equirec2Cube(h,w,d)
            self.c2e = Cube2Equirec(d,h,w)
            self.r2d = Radius2Depth(d)
            self.d2r = Depth2Radius(d)
        
    def forward(self,x,mode,s=2):
        assert mode in ['e2c','c2e','r2d','d2r']

        if mode in ['e2c','r2d']:
            h,w = x.shape[-2:]
            d = h // s
        else:
            d = x.shape[-1]
            h = d*s
            w = h//2

        if not (self.h,self.w,self.d) == (h,w,d):
            self.reinit(h,w,d)

        return getattr(self,mode)(x)
