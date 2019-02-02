import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# SENet's Module
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,with_variation=False):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if with_variation:
            out_planes = planes+1
        else:
            out_planes = planes
        self.conv2 = conv3x3(planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.se = SEModule(out_planes,reduction=16)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += self.downsample(x)
        out = self.relu(out)

        return out


# -----------------------
# Class: EmbeddingSENet
# Description: A SENet based embedding network for feature extraction.
#              if with_variation = True, a variational SENet would be constructed.
# -----------------------
class EmbeddingSENet(nn.Module):

    def __init__(self, block, layers, with_variation=True):
        self.inplanes = 64
        super(EmbeddingSENet, self).__init__()
        self.with_variation = with_variation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool1 = nn.AvgPool2d(56)
        self.avgpool2 = nn.AvgPool2d(28)
        self.avgpool3 = nn.AvgPool2d(14)
        self.avgpool4 = nn.AvgPool2d(7)
        self.expansion = block.expansion

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_class)


    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes,planes,with_variation=self.with_variation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.with_variation: # variational version
            feature1 = self.layer1(x) # [expansion*64+1,56,56]
            split_size = [self.expansion*64,1]
            feature1_mean,feature1_std = torch.split(feature1,split_size,dim=1)
            feature1_std = torch.sigmoid(feature1_std)
            feature1_std_ext = feature1_std.repeat(1,split_size[0],1,1)
            feature1 = feature1_mean + feature1_std_ext*torch.randn(feature1_mean.size(),device=feature1.get_device())
            feature1_avg = self.avgpool1(feature1)

            feature2 = self.layer2(feature1) #[expansion*128+1,28,28]
            split_size = [self.expansion*128,1]
            feature2_mean,feature2_std = torch.split(feature2,split_size,dim=1)
            feature2_std = torch.sigmoid(feature2_std)
            feature2_std_ext = feature2_std.repeat(1,split_size[0],1,1)
            feature2 = feature2_mean + feature2_std_ext*torch.randn(feature2_mean.size(),device=feature2.get_device())
            feature2_avg = self.avgpool2(feature2)

            feature3 = self.layer3(feature2) #[expansion*256+1,14,14]
            split_size = [self.expansion*256,1]
            feature3_mean,feature3_std = torch.split(feature3,split_size,dim=1)
            feature3_std = torch.sigmoid(feature3_std)
            feature3_std_ext = feature3_std.repeat(1,split_size[0],1,1)
            feature3 = feature3_mean + feature3_std_ext*torch.randn(feature3_mean.size(),device=feature3.get_device())
            feature3_avg = self.avgpool3(feature3)

            feature4 = self.layer4(feature3) #[expansion*512+1,7,7]
            split_size = [self.expansion*512,1]
            feature4_mean,feature4_std = torch.split(feature4,split_size,dim=1)
            feature4_std = torch.sigmoid(feature4_std)
            feature4_std_ext = feature4_std.repeat(1,split_size[0],1,1)
            feature4 = feature4_mean + feature4_std_ext*torch.randn(feature4_mean.size(),device=feature4.get_device())
            feature4_avg = self.avgpool4(feature4)

            feature1_std = feature1_std.view(feature1_std.size(0),-1)
            feature2_std = feature2_std.view(feature2_std.size(0),-1)
            feature3_std = feature3_std.view(feature3_std.size(0),-1)
            feature4_std = feature4_std.view(feature4_std.size(0),-1)

            std_mean = (torch.mean(feature1_std,1) + torch.mean(feature2_std,1) + torch.mean(feature3_std,1) + torch.mean(feature4_std,1))/4.0

        else: #standard version
            feature1 = self.layer1(x) # [expansion*64,56,56]
            feature2 = self.layer2(feature1) #[expansion*128,28,28]
            feature3 = self.layer3(feature2) #[expansion*256,14,14]
            feature4 = self.layer4(feature3) #[expansion*512,7,7]
            std_mean = torch.zeros(feature1.size(0),1,device = feature1.get_device())
            feature1_avg = self.avgpool1(feature1)
            feature2_avg = self.avgpool2(feature2)
            feature3_avg = self.avgpool3(feature3)
            feature4_avg = self.avgpool4(feature4)

        feature1_avg = feature1_avg.view(feature1_avg.size(0),-1)
        feature2_avg = feature2_avg.view(feature2_avg.size(0),-1)
        feature3_avg = feature3_avg.view(feature3_avg.size(0),-1)
        feature4_avg = feature4_avg.view(feature4_avg.size(0),-1)

        x = self.avgpool(feature4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)



        return feature1_avg,feature2_avg,feature3_avg,feature4_avg,std_mean,x
