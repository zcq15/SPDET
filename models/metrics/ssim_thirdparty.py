'''
 Code modified from https://github.com/Po-Hsun-Su/pytorch-ssim
'''

import torch
import numpy
import math

def __gaussian__(kernel_size, std, data_type=torch.float32):    
    gaussian = numpy.array([math.exp(-(x - kernel_size//2)**2/float(2*std**2)) for x in range(kernel_size)])
    gaussian /= numpy.sum(gaussian)
    return torch.tensor(gaussian, dtype=data_type)

def __create_kernel__(kernel_size, data_type=torch.float32, channels=3, std=1.5):
    gaussian1d = __gaussian__(kernel_size, std).unsqueeze(1)
    gaussian2d = torch.mm(gaussian1d, gaussian1d.t())\
        .type(data_type)\
        .unsqueeze(0)\
        .unsqueeze(0)
    window = gaussian2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return window

def __ssim_gaussian__(prediction, groundtruth, kernel, kernel_size, channels=3, pad=None):
    prediction_mean = torch.nn.functional.conv2d(pad(prediction), kernel, padding=0, groups=channels)
    groundtruth_mean = torch.nn.functional.conv2d(pad(groundtruth), kernel, padding=0, groups=channels)

    prediction_mean_squared = prediction_mean.pow(2)
    groundtruth_mean_squared = groundtruth_mean.pow(2)
    prediction_mean_times_groundtruth_mean = prediction_mean * groundtruth_mean

    prediction_sigma_squared = torch.nn.functional.conv2d(pad(prediction * prediction), kernel, padding=0, groups=channels)\
        - prediction_mean_squared
    groundtruth_sigma_squared = torch.nn.functional.conv2d(pad(groundtruth * groundtruth), kernel, padding=0, groups=channels)\
        - groundtruth_mean_squared
    prediction_groundtruth_covariance = torch.nn.functional.conv2d(pad(prediction * groundtruth), kernel, padding=0, groups=channels)\
        - prediction_mean_times_groundtruth_mean

    C1 = 0.01**2 # assume that images are in the [0, 1] range
    C2 = 0.03**2 # assume that images are in the [0, 1] range

    return (
        ( # numerator
            (2 * prediction_mean_times_groundtruth_mean + C1) # luminance term
            * (2 * prediction_groundtruth_covariance + C2) # structural term
        )
        / # division
        ( # denominator
            (prediction_mean_squared + groundtruth_mean_squared + C1) # luminance term
            * (prediction_sigma_squared + groundtruth_sigma_squared + C2) # structural term
        )
    ) 

def ssim_gaussian(prediction, groundtruth, kernel_size=11, std=1.5, pad=None):
    (_, channels, _, _) = prediction.size()
    kernel = __create_kernel__(kernel_size, data_type=prediction.type(),\
        channels=channels, std=std)
    
    if prediction.is_cuda:
        kernel = kernel.to(prediction.get_device())
    kernel = kernel.type_as(prediction)
    
    return __ssim_gaussian__(prediction, groundtruth, kernel, kernel_size, channels, pad=pad)

def ssim_box(prediction, groundtruth, kernel_size=3, pad=None):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    channels = prediction.shape[1]
    kernel = torch.ones([channels,1,kernel_size,kernel_size],dtype=torch.float32,device=prediction.device)/(kernel_size*kernel_size)

    prediction_mean = torch.nn.functional.conv2d(pad(prediction), kernel, padding=0, groups=channels)
    groundtruth_mean = torch.nn.functional.conv2d(pad(groundtruth), kernel, padding=0, groups=channels)
    prediction_groundtruth_mean = prediction_mean * groundtruth_mean
    prediction_mean_squared = prediction_mean.pow(2)
    groundtruth_mean_squared = groundtruth_mean.pow(2)

    prediction_sigma = torch.nn.functional.conv2d(pad(prediction * prediction), kernel, padding=0, groups=channels) - prediction_mean_squared
    groundtruth_sigma = torch.nn.functional.conv2d(pad(groundtruth * groundtruth), kernel, padding=0, groups=channels) - groundtruth_mean_squared
    correlation = torch.nn.functional.conv2d(pad(prediction * groundtruth), kernel, padding=0, groups=channels) - prediction_groundtruth_mean

    numerator = (2 * prediction_groundtruth_mean + C1) * (2 * correlation + C2)
    denominator = (prediction_mean_squared + groundtruth_mean_squared + C1)\
        * (prediction_sigma + groundtruth_sigma + C2)
    ssim = numerator / denominator
    return ssim

def ssim_loss(prediction, groundtruth, kernel_size=5, std=1.5, mode='gaussian', pad=None):
    if mode == 'gaussian':
        return ssim_gaussian(prediction, groundtruth, kernel_size=kernel_size, std=std, pad=pad)
    elif mode == 'box':
        return ssim_box(prediction, groundtruth, kernel_size=kernel_size, pad=pad)