import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, dropout_keep_prob = 0.5):
        super(GoogLeNet, self).__init__()

        conv_block = Unit2d
        inception_block = Inception

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d((7, 7))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.fc = nn.Linear(1024, num_classes)

        # if self.pre_train_file:
        #     self.load_org_weights()
        # else:
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

    def load_org_weights(self):
        tic = time.time()
        model_dict = self.state_dict()
        pre_dict = torch.load(self.pre_train_file)
        if 'state_dict' in pre_dict.keys():
            pre_dict = pre_dict['state_dict']
        for name in model_dict.keys():
            if model_dict[name].shape == pre_dict[name].shape:
                model_dict[name] = pre_dict[name]
            else:
                print('size mismatch for %s, expect (%s), but got (%s).'
                      % (name, ','.join([str(x) for x in model_dict[name].shape]),
                         ','.join([str(x) for x in pre_dict[name].shape])))

        self.load_state_dict(model_dict)

        print('Load pre-trained weight %s in %.4f sec.' % (self.pre_train_file, time.time()-tic))

        for k, v in pre_dict.items():
            for name in model_dict.keys():
                if name == k:
                    if model_dict[name].shape == v.shape:
                        model_dict[name] = v
                    else:
                        print('size mismatch for %s, expect (%s), but got (%s).'
                              %(name, ','.join([str(x) for x in model_dict[name].shape]),','.join([str(x) for x in v.shape])))
                        continue

        self.load_state_dict(model_dict)

        print('Load pre-trained weight %s in %.4f sec.' % (self.pre_train_file, time.time()-tic))


class Inception(nn.Module):
    __constants__ = ['branch2', 'branch3', 'branch4']

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 conv_block=None, groups=1):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = Unit2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1, groups=groups)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1, groups=groups),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1, groups=groups)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1, groups=groups),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1, groups=groups)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1, groups=groups)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class Unit2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Unit2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    from utils.thop import profile, clever_format
    d_in = torch.rand(16, 3, 224, 224)
    m = GoogLeNet()
    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
