import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import torch.utils.model_zoo as model_zoo


model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3d(nn.Module):
    def __init__(self, in_planes, out_planes,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0):
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
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(SepUnit3d, self).__init__()
        self.s_conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size), stride=(1,stride,stride), padding=(0,padding,padding), bias=False)
        self.s_bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.s_relu = nn.ReLU()

        self.t_conv = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.t_bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.t_relu = nn.ReLU()

    def forward(self, x):
        x = self.s_conv(x)
        x = self.s_bn(x)
        x = self.s_relu(x)

        x = self.t_conv(x)
        x = self.t_bn(x)
        x = self.t_relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(Inception, self).__init__()

        self.branch1 = Unit3d(in_channels, output_channels[0])

        self.branch2 = nn.Sequential(
            Unit3d(in_channels, output_channels[1]),
            SepUnit3d(output_channels[1], output_channels[2], kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            Unit3d(in_channels, output_channels[3]),
            SepUnit3d(output_channels[3], output_channels[4], kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                 stride=(1, 1, 1), padding=0),
            Unit3d(in_channels, output_channels[5])
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        out = torch.cat(outputs, 1)
        return out


class S3D(nn.Module):

    def __init__(self, pre_train=False, num_classes=400):
        super(S3D, self).__init__()
        inception_block = Inception

        self.conv1 = Unit3d(3, 64, kernel_size=(7,7,7), stride=(2,2,2), padding=(3,3,3))
        self.maxpool1 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        self.conv2 = Unit3d(64, 64)
        self.conv3 = Unit3d(64, 192, kernel_size=(3,3,3), padding=1)
        self.maxpool2 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)

        self.inception3a = inception_block(192, [64, 96, 128, 16, 32, 32])
        self.inception3b = inception_block(256, [128, 128, 192, 32, 96, 64])
        self.maxpool3 = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)

        self.inception4a = inception_block(480, [192, 96, 208, 16, 48, 64])
        self.inception4b = inception_block(512, [160, 112, 224, 24, 64, 64])
        self.inception4c = inception_block(512, [128, 128, 256, 24, 64, 64])
        self.inception4d = inception_block(512, [112, 144, 288, 32, 64, 64])
        self.inception4e = inception_block(528, [256, 160, 320, 32, 128, 128])
        self.maxpool4 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)

        self.inception5a = inception_block(832, [256, 160, 320, 32, 128, 128])
        self.inception5b = inception_block(832, [384, 192, 384, 48, 128, 128])

        self.avgpool = nn.AvgPool3d((2,7, 7))
        self.fc = nn.Linear(1024, num_classes)

        if pre_train:
            self.load_org_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def load_org_weights(self):
        pre_dict = model_zoo.load_url(model_urls['googlenet'])
        tic = time.time()
        model_dict = self.state_dict()
        if 'state_dict' in pre_dict.keys():
            pre_dict = pre_dict['state_dict']
        for name in model_dict.keys():
            is_null = True
            try:
                if model_dict[name].shape == pre_dict[name].shape:
                    model_dict[name] = pre_dict[name]
                    is_null = False
                else:
                    if model_dict[name].shape[0:2] + model_dict[name].shape[3:] == pre_dict[name].shape:
                        t = model_dict[name].shape[2]
                        model_dict[name] = pre_dict[name].repeat(t,1,1,1,1).permute(1,2,0,3,4)/t
                        is_null=False
                    else:
                        print('size mismatch for %s, expect (%s), but got (%s).'
                              % (name, ','.join([str(x) for x in model_dict[name].shape]),
                                 ','.join([str(x) for x in pre_dict[name].shape])))
                continue
            except KeyError:
                for k, v in pre_dict.items():
                    if re.sub('s_', '', name) == k:
                        if model_dict[name].shape == v.shape:
                            model_dict[name] = v
                            is_null = False
                        else:
                            if model_dict[name].shape[0:2] + model_dict[name].shape[3:] == v.shape:
                                t = model_dict[name].shape[2]
                                model_dict[name] = v.repeat(t, 1, 1, 1, 1).permute(1, 2, 0, 3, 4) / t
                                is_null = False
                            else:
                                print('size mismatch for %s, expect (%s), but got (%s).'
                                      % (name, ','.join([str(x) for x in model_dict[name].shape]),
                                         ','.join([str(x) for x in v.shape])))

            if is_null and 'bn' not in name:
                print('Do not load %s' % name)

        self.load_state_dict(model_dict)

        print('Load pre-trained weight in %.4f sec.' % (time.time() - tic))


if __name__ == '__main__':
    m = S3D(True)
    from utils.thop import profile, clever_format
    d_in = torch.rand(1, 3, 16, 224, 224)
    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
