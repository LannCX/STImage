import torch
import torch.nn as nn


class VGGM(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # self.pool = nn.MaxPool2d((7,7))
        self.fc6 = nn.Linear(512*7*7, 4096)
        self.fc7 = nn.Linear(4096,2048)
        self.logit = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inp):
        x = self.feature(inp)
        x = torch.flatten(x, 1)
        # x = self.pool(x).squeeze(2).squeeze(2)
        x = self.fc6(x)
        # x = self.dropout(x)
        x = self.fc7(x)
        x = self.dropout(x)
        x = self.logit(x)

        return x


if __name__ == '__main__':
    from torchvision.models.alexnet import alexnet
    from torchvision.models.mobilenet import mobilenet_v2
    from torchvision.models.resnet import resnet50
    m = resnet50(num_classes=101)
    # m = mobilenet_v2(num_classes=101)
    # m = VGGM(num_classes=101)
    from utils.thop import profile, clever_format
    d_in = torch.rand(16, 3, 224, 224)
    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
