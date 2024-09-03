import re
import torch.utils.model_zoo as model_zoo
from arch.conv import *


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

size = None


def conv3x3(in_planes, out_planes, stride=1, t_size=4):
    """3x3 convolution with padding"""
    return Unit4d(in_planes, out_planes, stride=stride, size=t_size)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_4d=False):
        super(BasicBlock, self).__init__()
        global size
        self.conv1 = conv3x3(inplanes, planes, stride, t_size=size) \
            if is_4d else CoCov(inplanes, planes, stride=stride, t_size=size)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, t_size=size) \
            if is_4d else CoCov(planes, planes, t_size=size)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_4d=False):
        super(Bottleneck, self).__init__()
        global size
        self.conv1 = CoCov(inplanes, planes, kernel_size_h=3, kernel_size_l=1, t_size=size, padding=1) if is_4d \
            else nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CoCov(planes, planes, stride=stride, t_size=size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, t_size=4):
        global size
        size = t_size
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = CoCov(3, 64, kernel_size_l=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = GridMaxPool(kernel_size=3, stride_l=2, t_size=size)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, is_4d=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_4d=True)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.5)
        self.final_conv = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=size)
        # self.max_pool = nn.MaxPool2d(size, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_org_weights(self, pre_dict):
        tic = time.time()
        model_dict = self.state_dict()
        if 'state_dict' in pre_dict.keys():
            pre_dict = pre_dict['state_dict']
        for name in model_dict.keys():
            if 'num_batches_tracked' in name:
                continue
            is_null = True
            try:
                if model_dict[name].shape == pre_dict[name].shape:
                    model_dict[name] = pre_dict[name]
                    is_null = False
                else:
                    print('size mismatch for %s, expect (%s), but got (%s).'
                          % (name, ','.join([str(x) for x in model_dict[name].shape]),
                             ','.join([str(x) for x in pre_dict[name].shape])))
                continue
            except KeyError:
                for k, v in pre_dict.items():
                    if re.sub('kernels\.', '', name)==k:
                        model_dict[name] = v.data
                        is_null = False
                        break
                    elif re.sub('s_conv\.kernels\.\d{1,2}\.', '', name) == k or \
                        re.sub('kernels\.\d{1,2}\.', '', name)==k:
                        if model_dict[name].shape == v.shape:
                            model_dict[name] = v
                            is_null = False
                        elif model_dict[name].shape[1] != v.shape[1]:
                            print('size mismatch for %s, expect (%s), but got (%s).'
                                  % (name, ','.join([str(x) for x in model_dict[name].shape]),
                                     ','.join([str(x) for x in v.shape])))
                            continue
                        break
            if is_null:
                print('Do not load %s' % name)

        self.load_state_dict(model_dict)

        print('Load pre-trained weight in %.4f sec.' % (time.time() - tic))

    def _make_layer(self, block, planes, blocks, stride=1, is_4d=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, is_4d))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        # x = self.max_pool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x.squeeze(2).squeeze(2)


class ResNet4D18(nn.Module):
    def __init__(self, cfg):
        super(ResNet4D18, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2],
                            num_classes=cfg.MODEL.NUM_CLASSES,
                            t_size=cfg.DATASET.T_SIZE)
        self.model.load_org_weights(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, x):
        return self.model(x)


class ResNet4D34(nn.Module):
    def __init__(self, cfg):
        super(ResNet4D34, self).__init__()
        self.model = ResNet(BasicBlock, [3, 4, 6, 3],
                            num_classes=cfg.MODEL.NUM_CLASSES,
                            t_size=cfg.DATASET.T_SIZE)
        self.model.load_org_weights(model_zoo.load_url(model_urls['resnet34']))

    def forward(self, x):
        return self.model(x)


class ResNet4D50(nn.Module):
    def __init__(self, cfg):
        super(ResNet4D50, self).__init__()
        block = Bottleneck
        self.model = ResNet(block, [3, 4, 6, 3],
                            num_classes=cfg.MODEL.NUM_CLASSES,
                            t_size=cfg.DATASET.T_SIZE)
        self.model.load_org_weights(model_zoo.load_url(model_urls['resnet50']))

    def forward(self, x):
        return self.model(x)


class ResNet4D101(nn.Module):
    def __init__(self, cfg):
        super(ResNet4D101, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 23, 3],
                            num_classes=cfg.MODEL.NUM_CLASSES,
                            t_size=cfg.DATASET.T_SIZE)
        self.model.load_org_weights(model_zoo.load_url(model_urls['resnet101']))

    def forward(self, x):
        return self.model(x)


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg', default='../../experiments/something-something.yaml',
                        help='experiment configure file name',
                        # required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    from cfg.coconv_cfg import CoConvConfig as Config
    args = parse_args()
    cfg = Config(args).getcfg()
    d_in = torch.rand(1, 3 * cfg.DATASET.K ** 2,
                            cfg.DATASET.T_SIZE*cfg.DATASET.CROP_SIZE[0]//cfg.DATASET.K,
                            cfg.DATASET.T_SIZE*cfg.DATASET.CROP_SIZE[0]//cfg.DATASET.K)
    m = ResNet4D50(cfg)
    from utils.thop import profile, clever_format
    # from torchvision.models.resnet import resnet101, resnet50
    # m = resnet50()
    # d_in = torch.rand(16, 3, 224, 224)

    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
