from torchvision import models
from torch import nn
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
from shapely import wkt, box
import rasterio
from PIL import Image
from collections import Counter
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class LinearRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels=in_channels, out_channels=out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class DamageClassifierCC(nn.Module):
    def __init__(self):
        super(DamageClassifierCC, self).__init__()

        resnet18 = models.resnet18(weights="DEFAULT")
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = ConvRelu(512, 1024)
        self.linear = nn.Linear(1024, 4)

    def forward(self, x_pre, x_post):
        x = torch.cat([x_pre, x_post], 1)
        x = self.resnet18(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class DamageClassifierPO(nn.Module):
    def __init__(self):
        super(DamageClassifierPO, self).__init__()

        resnet18 = models.resnet18(weights="DEFAULT")
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = ConvRelu(512, 1024)
        self.linear = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x 
        
class DamageClassifierTTC(nn.Module):
    def __init__(self):
        super(DamageClassifierTTC, self).__init__()

        resnet18 = models.resnet18(weights="DEFAULT")
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = ConvRelu(1024, 2048)
        self.linear = nn.Linear(2048, 4)

    def forward(self, x_pre, x_post):
        pre_features = self.resnet18(x_pre)
        post_features = self.resnet18(x_post)
        x = torch.cat([pre_features, post_features], 1)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x 

class DamageClassifierTTS(nn.Module):
    def __init__(self):
        super(DamageClassifierTTS, self).__init__()

        resnet18 = models.resnet18(weights="DEFAULT")
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = ConvRelu(512, 1024)
        self.linear = nn.Linear(1024, 4)

    def forward(self, x_pre, x_post):
        pre_features = self.resnet18(x_pre)
        post_features = self.resnet18(x_post)
        x = torch.subtract(post_features, pre_features)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x 