"""
S3D
from https://github.com/kylemin/S3D/blob/master/model.py
"""
import torch
from torch import nn
import torch.nn.functional as F


class S3D(nn.Module):
    def __init__(self, num_class=101):
        super(S3D, self).__init__()
        self.base = nn.Sequential(
            SepUnit3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            Unit3d(64, 64, kernel_size=1, stride=1),
            SepUnit3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            Mixed_3b(),
            Mixed_3c(),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0)),
            Mixed_5b(),
            Mixed_5c(),
        )
        self.fc = nn.Sequential(nn.Conv3d(1024, num_class, kernel_size=1, stride=1, bias=True),)

    def forward(self, x):
        y = self.base(x)
        y = F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)
        y = self.fc(y)
        y = y.view(y.size(0), y.size(1), y.size(2))
        logits = torch.mean(y, 2)

        return logits


class Unit3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(Unit3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SepUnit3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepUnit3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size), stride=(1,stride,stride), padding=(0,padding,padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            Unit3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Unit3d(192, 96, kernel_size=1, stride=1),
            SepUnit3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Unit3d(192, 16, kernel_size=1, stride=1),
            SepUnit3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            Unit3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            Unit3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Unit3d(256, 128, kernel_size=1, stride=1),
            SepUnit3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Unit3d(256, 32, kernel_size=1, stride=1),
            SepUnit3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            Unit3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            Unit3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Unit3d(480, 96, kernel_size=1, stride=1),
            SepUnit3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Unit3d(480, 16, kernel_size=1, stride=1),
            SepUnit3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            Unit3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            Unit3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Unit3d(512, 112, kernel_size=1, stride=1),
            SepUnit3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Unit3d(512, 24, kernel_size=1, stride=1),
            SepUnit3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            Unit3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            Unit3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Unit3d(512, 128, kernel_size=1, stride=1),
            SepUnit3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Unit3d(512, 24, kernel_size=1, stride=1),
            SepUnit3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            Unit3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            Unit3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Unit3d(512, 144, kernel_size=1, stride=1),
            SepUnit3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Unit3d(512, 32, kernel_size=1, stride=1),
            SepUnit3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            Unit3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            Unit3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Unit3d(528, 160, kernel_size=1, stride=1),
            SepUnit3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Unit3d(528, 32, kernel_size=1, stride=1),
            SepUnit3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            Unit3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            Unit3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Unit3d(832, 160, kernel_size=1, stride=1),
            SepUnit3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Unit3d(832, 32, kernel_size=1, stride=1),
            SepUnit3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            Unit3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            Unit3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Unit3d(832, 192, kernel_size=1, stride=1),
            SepUnit3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Unit3d(832, 48, kernel_size=1, stride=1),
            SepUnit3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            Unit3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

if __name__ == '__main__':
    from utils.nn_utils import get_model_summary

    # data = torch.load()
    m = S3D()
    from utils.thop import profile, clever_format
    d_in = torch.rand(1, 3, 16, 224, 224)
    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
