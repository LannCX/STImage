import torch.nn as nn
import math
import time
import torch.utils.model_zoo as model_zoo
from arch.conv import *
from utils.nn_utils import video_to_stimg, stimg_to_video

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
                     padding=1*dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, t_size=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.s_conv2 = nn.Conv2d(planes, planes//2, kernel_size=3, stride=stride, dilation=t_size, padding=t_size, bias=False)
        self.t_conv2 = nn.Conv2d(planes, planes//2, kernel_size=3, stride=stride, dilation=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.t_size = t_size

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        b1 = self.s_conv2(out)
        # if self.stride==2:
        #     b, c, h, w = b1.shape
        #     inds = [i*self.t_size+j for i in range(0, w//self.t_size, 2) for j in range(self.t_size)]
        #     b1 = b1[:, :, inds, :][:, :, :, inds]
        b2 = self.t_conv2(out)
        # if self.stride == 2:
        #     b, c, h, w = b2.shape
        #     inds = [i * self.t_size + j for i in range(0, w // self.t_size, 2) for j in range(self.t_size)]
        #     b2 = b2[:, :, inds, :][:, :, :, inds]

        out = torch.cat([b1, b2], 1)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            # if self.stride == 2:
            #     b, c, h, w = residual.shape
            #     inds = [i * self.t_size + j for i in range(0, w // self.t_size, 2) for j in range(self.t_size)]
            #     residual = residual[:, :, inds, :][:, :, :, inds]

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, t_size=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.t_size = t_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], t_size=t_size)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, t_size=t_size)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, t_size=t_size)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, t_size=t_size)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.logits = nn.Conv2d(512 * block.expansion, num_classes, t_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, t_size=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, t_size=t_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, t_size=t_size))

        return nn.Sequential(*layers)

    def load_org_weights(self, pre_dict):
        tic = time.time()
        model_dict = self.state_dict()
        if 'state_dict' in pre_dict.keys():
            pre_dict = pre_dict['state_dict']
        for name, val in pre_dict.items():
            n = val.shape[0]
            if name in model_dict.keys() and model_dict[name].shape==val.shape:
                model_dict[name] = val
            elif 'conv' in name:
                last_name = name.split('.')[-2]
                new_name = name.replace(last_name, 's_'+last_name)
                model_dict[new_name] = val[:n//2,...]
                new_name = name.replace(last_name, 't_'+last_name)
                model_dict[new_name] = val[n//2:,...]
            else:
                print('Do not load %s' % name)
        # for name in model_dict.keys():
        #     if 'num_batches_tracked' in name:
        #         continue
        #     try:
        #         if model_dict[name].shape == pre_dict[name].shape:
        #             model_dict[name] = pre_dict[name]
        #         else:
        #             print('size mismatch for %s, expect (%s), but got (%s).'
        #                   % (name, ','.join([str(x) for x in model_dict[name].shape]),
        #                      ','.join([str(x) for x in pre_dict[name].shape])))
        #         continue
        #     except KeyError:
        #         print('Do not load %s' % name)

        self.load_state_dict(model_dict)
        print('Load pre-trained weight in %.4f sec.' % (time.time() - tic))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.reshape((-1, self.t_size**2) + x.shape[-3:])
        x = torch.stack([video_to_stimg(vid, self.t_size) for vid in x])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.stack([stimg_to_video(img, self.t_size) for img in x])
        x = x.reshape((-1,)+x.shape[-3:])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_org_weights(model_zoo.load_url(model_urls['resnet50']))
    return model


if __name__ == '__main__':
    from utils.thop import profile, clever_format
    # from torchvision.models.resnet import resnet50

    d_in = torch.rand(9, 3, 224, 224)
    m = resnet50(True, t_size=3, num_classes=174)
    # m = resnet50(num_classes=174)
    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
