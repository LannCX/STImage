import re
from arch.conv import *
from utils.nn_utils import load_checkpoint


class Inception(nn.Module):
    def __init__(self, in_channels, output_channels, size, groups=1, stride=1):
        super(Inception, self).__init__()
        conv_block = Unit2d

        self.branch1 = conv_block(in_channels, output_channels[0], kernel_size_h=1, kernel_size_l=1,
                                  t_size=size, groups=groups, stride=stride)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, output_channels[1], kernel_size_h=1, kernel_size_l=1,
                       t_size=size, stride=stride, groups=groups),
            conv_block(output_channels[1], output_channels[2], kernel_size_h=1, kernel_size_l=3,
                       t_size=size, groups=groups),
            # conv_block(output_channels[2], output_channels[2], kernel_size_h=3, kernel_size_l=1,
            #            t_size=size, padding=1, groups=groups)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, output_channels[3], kernel_size_h=1, kernel_size_l=1,
                       t_size=size, stride=stride, groups=groups),
            # conv_block(output_channels[3], output_channels[4], kernel_size_h=1, kernel_size_l=3,
            #            t_size=size, groups=groups),
            conv_block(output_channels[3], output_channels[4], kernel_size_h=3, kernel_size_l=1,
                       t_size=size, padding=1, groups=groups)
        )

        self.branch4 = nn.Sequential(
            # max_pool_3x3x3x3_s1t1(size),
            MaxPool2dSamePadding(t_size=size, kernel_size=3, stride=1),
            conv_block(in_channels, output_channels[5], kernel_size_h=1, kernel_size_l=1,
                       t_size=size, stride=stride, groups=groups)
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
        out = torch.cat(outputs, 1)
        return out


class Unit2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation_fn=None,
                 use_norm=True,
                 use_bias=False,
                 **kwargs):
        super(Unit2d, self).__init__()
        self._use_bias = use_bias
        self._use_batch_norm = use_norm
        if activation_fn is None:
            self._activation_fn = nn.ReLU(inplace=True)
        else:
            self._activation_fn = activation_fn
        self.conv = CoCov(in_channels, out_channels, bias=self._use_bias, **kwargs)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
            # self.gn = nn.GroupNorm(num_channels=out_channels, num_groups=4)

    def forward(self, x):
        x = self.conv(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionV1(nn.Module):

    def __init__(self, cfg):
        super(InceptionV1, self).__init__()
        num_classes = cfg.MODEL.NUM_CLASSES
        dropout_keep_prob = cfg.MODEL.DROPOUT

        self.ssl = cfg.USE_SSL
        self.size = cfg.DATASET.T_SIZE
        self.pre_train_file = cfg.MODEL.PRETRAINED

        conv_block = Unit2d
        inception_block = Inception

        self.conv1 = conv_block(3, 64, kernel_size_l=7,  kernel_size_h=1, stride=2, t_size=self.size)
        self.maxpool1 = MaxPool2dSamePadding(t_size=self.size, kernel_size=3, stride=2)
        self.conv2 = conv_block(64, 64, kernel_size_l=1, t_size=self.size)
        self.conv3 = conv_block(64, 192, kernel_size_l=3, t_size=self.size)
        self.maxpool2 = MaxPool2dSamePadding(t_size=self.size, kernel_size=3, stride=2)

        self.inception3a = inception_block(192, [64, 96, 128, 16, 32, 32], size=self.size)
        self.inception3b = inception_block(256, [128, 128, 192, 32, 96, 64], size=self.size)
        self.maxpool3 = MaxPool2dSamePadding(t_size=self.size, kernel_size=3, stride=2)

        self.inception4a = inception_block(480, [192, 96, 208, 16, 48, 64], size=self.size)
        self.inception4b = inception_block(512, [160, 112, 224, 24, 64, 64], size=self.size)

        self.inception4c = inception_block(512, [128, 128, 256, 24, 64, 64], size=self.size)
        self.inception4d = inception_block(512, [112, 144, 288, 32, 64, 64], size=self.size)
        self.inception4e = inception_block(528, [256, 160, 320, 32, 128, 128], size=self.size)
        self.maxpool4 = GridMaxPool(t_size=self.size, kernel_size=3, stride=2)

        self.inception5a = inception_block(832, [256, 160, 320, 32, 128, 128], size=self.size)
        self.inception5b = inception_block(832, [384, 192, 384, 48, 128, 128], size=self.size)

        self.avgpool = nn.AvgPool2d((7, 7))  # default stride value is kernel_size
        self.dropout = nn.Dropout(dropout_keep_prob)

        self.logits = nn.Conv2d(1024, num_classes, self.size)

        if self.pre_train_file:
            self.load_org_weights()

    def forward(self, x):
        # input: N x (3xKxK) x (t_size x H//K ) x (t_size x W//K ), t_size=12, p_size=H//K
        x = self.conv1(x)  # N x 64 x 896 x 896, t_size=8, p_size=112
        x = self.maxpool1(x)  # N x 64 x 448 x 448, t_size=8, p_size=56
        x = self.conv2(x)  # N x 192 x 448 x 448, t_size=8, p_size=56
        x = self.conv3(x)  # N x 192 x 448 x 448, t_size=8, p_size=56
        x = self.maxpool2(x)  # N x 192 x 224 x 224, t_size=8, p_size=28

        x = self.inception3a(x)  # N x 256 x 112 x 112, t_size=8, p_size=14
        x = self.inception3b(x)  # N x 480 x 112 x 112, t_size=8, p_size=14
        x = self.maxpool3(x)  # N x 480 x 56 x 56, t_size=8, p_size=7

        x = self.inception4a(x)  # N x 512 x 56 x 56, t_size=8, p_size=7
        x = self.inception4b(x)  # N x 512 x 56 x 56, t_size=8, p_size=7
        x = self.inception4c(x)  # N x 528 x 56 x 56, t_size=8, p_size=7
        x = self.inception4d(x)  # N x 528 x 56 x 56, t_size=8, p_size=7
        x = self.inception4e(x)  # N x 832 x 56 x 56, t_size=8, p_size=7
        x = self.maxpool4(x)  # N x 832 x 28 x 28, t_size=4, p_size=7

        x = self.inception5a(x)  # N x 832 x 28 x 28, t_size=4, p_size=7
        feat = self.inception5b(x)  # N x 1024 x 28 x 28, t_size=4, p_size=7
        x = self.avgpool(feat)  # N x 1024 x 4 x 4, t_size=4, p_size=1
        x = self.dropout(x)

        cls_logits = self.logits(x)
        return cls_logits.squeeze(2).squeeze(2)

    def load_org_weights(self):
        tic = time.time()
        model_dict = self.state_dict()
        pre_dict = torch.load(self.pre_train_file)
        if 'state_dict' in pre_dict.keys():
            pre_dict = pre_dict['state_dict']
        for name in model_dict.keys():
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
                    if re.sub('kernels\.', '', name) == k:
                        model_dict[name] = v.data
                        is_null = False
                        break
                    elif re.sub('kernels\.\d{1,2}\.', '', name) == k:
                        if model_dict[name].shape == v.shape:
                            model_dict[name] = v
                            is_null = False
                        elif model_dict[name].shape[1] != v.shape[1]:
                            print('size mismatch for %s, expect (%s), but got (%s).'
                                  % (name, ','.join([str(x) for x in model_dict[name].shape]),
                                     ','.join([str(x) for x in v.shape])))
                            continue
                        else:
                            num = int(name[-8])
                            model_dict[name] = v.data[:, :, num // 3, num % 3].unsqueeze(2).unsqueeze(3)
                            is_null = False
                        break
            if is_null:
                print('Do not load %s' % name)

        self.load_state_dict(model_dict)

        print('Load pre-trained weight %s in %.4f sec.' % (self.pre_train_file, time.time()-tic))

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
    d_in = torch.rand(1, 3,
                      cfg.DATASET.T_SIZE*cfg.DATASET.CROP_SIZE[0],
                      cfg.DATASET.T_SIZE*cfg.DATASET.CROP_SIZE[0])
    m = InceptionV1(cfg)
    from utils.thop import profile, clever_format

    # d_in = torch.rand(1, 3, 16, 224, 224)
    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
