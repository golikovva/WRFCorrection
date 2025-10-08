import numpy as np
from torch import nn
from collections import OrderedDict


# from correction.config import cfg
# import cv2
# import os.path as osp
# import os
# from correction.data.mask import read_mask_file

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            if '3d' in layer_name:
                conv = nn.Conv3d(in_channels=v[0], out_channels=v[1],
                                 kernel_size=v[2], stride=v[3],
                                 padding=v[4])
            else:
                conv = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                 kernel_size=v[2], stride=v[3],
                                 padding=v[4])
            layers.append((layer_name, conv))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'upsample' in layer_name:
            upsample = nn.Upsample(scale_factor=v[5])
            layers.append(('upsample_' + layer_name, upsample))
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                         kernel_size=v[2], stride=v[3],
                                         padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))
