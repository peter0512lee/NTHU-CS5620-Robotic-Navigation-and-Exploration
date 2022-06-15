# import necessacry libraries and set parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import models

import numpy as np


class Vgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16, self).__init__()
        self.vggnet = models.vgg16(pretrained)
        # Remove fully connected layer to save memory.
        del(self.vggnet.classifier)
        features = list(self.vggnet.features)
        self.layers = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.layers):
            x = model(x)
            if ii in [3, 8, 15, 22, 29]:
                # (64,256,256),(128,128,128),(256,64,64),(512,32,32),(512,16,16)
                results.append(x)
        return results


# vgg_model = Vgg16()
# vgg_model = vgg_model.cuda()
# print(vgg_model.layers)


class DeConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        output = self.up(x)
        output = self.conv(output)
        return output


class EncoderDecoder(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = Vgg16().cuda()
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = DeConv2d(512, 512, kernel_size=3,
                                stride=1, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = DeConv2d(512, 256, kernel_size=3,
                                stride=1, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = DeConv2d(256, 128, kernel_size=3,
                                stride=1, padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = DeConv2d(128, 64, kernel_size=3,
                                stride=1, padding=1, dilation=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        pre_output = self.pretrained_net(x)
        output = self.bn1(
            self.relu(self.deconv1(pre_output[4])))  # (512,32,32)
        output = self.bn2(self.relu(self.deconv2(output)))  # (256,64,64)
        output = self.bn3(self.relu(self.deconv3(output)))  # (128,128,128)
        output = self.bn4(self.relu(self.deconv4(output)))  # (64,256,256)
        output = self.classifier(output)
        return output
