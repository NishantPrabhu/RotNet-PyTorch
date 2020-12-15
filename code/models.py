
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import models


BACKBONES = {
    'resnet18': {'net': models.resnet18, 'dim': 512},
    'resnet34': {'net': models.resnet34, 'dim': 512},
    'resnet50': {'net': models.resnet50, 'dim': 2048},
    'resnet101': {'net': models.resnet101, 'dim': 2048}
}


class Encoder(nn.Module):

    def __init__(self, name='resnet18', zero_init_residual=False):
        super(Encoder, self).__init__()
        assert name in BACKBONES.keys(), 'name should be one of (resnet18, resnet34, resnet50, resnet101)'

        # Initial layers
        conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu1 = nn.ReLU()

        # Backbone
        resnet = BACKBONES[name]['net'](pretrained=False)
        layers = list(resnet.children())
        self.backbone = nn.Sequential(conv0, bn1, relu1, *layers[4:len(layers)-1])
        self.backbone_dim = BACKBONES[name]['dim']

        # Weight initialization
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Reference: https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.backbone.modules():
                if isinstance(m, models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        return out


class RotnetClassifier(nn.Module):

    def __init__(self, in_dim=512, n_classes=4):
        super(RotnetClassifier, self).__init__()
        self.W1 = nn.Linear(in_dim, in_dim)
        self.BN1 = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        out = self.W2(self.relu(self.BN1(self.W1(x))))
        out = F.log_softmax(out, dim=-1)
        return out


class LinearClassifier(nn.Module):

    def __init__(self, in_dim=512, n_classes=10):
        super(LinearClassifier, self).__init__()
        self.W1 = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return F.log_softmax(self.W1(x), dim=-1)