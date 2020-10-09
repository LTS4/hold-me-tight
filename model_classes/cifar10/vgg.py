'''VGG11/13/16/19 in Pytorch.'''
import math
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, batch_norm=False, num_channels=3, num_classes=10):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.batch_norm = batch_norm
        self.num_channels = num_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.num_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG11(VGG):
    def __init__(self, num_channels=3, num_classes=10):
        super(VGG11, self).__init__(vgg_name='VGG11', num_channels=num_channels, num_classes=num_classes)


class VGG11_bn(VGG):
    def __init__(self, num_channels=3, num_classes=10):
        super(VGG11_bn, self).__init__(vgg_name='VGG11', num_channels=num_channels, num_classes=num_classes,
                                       batch_norm=True)


class VGG13(VGG):
    def __init__(self, num_channels=3, num_classes=10):
        super(VGG13, self).__init__(vgg_name='VGG13', num_channels=num_channels, num_classes=num_classes)


class VGG13_bn(VGG):
    def __init__(self, num_channels=3, num_classes=10):
        super(VGG13_bn, self).__init__(vgg_name='VGG13', num_channels=num_channels, num_classes=num_classes,
                                       batch_norm=True)


class VGG16(VGG):
    def __init__(self, num_channels=3, num_classes=10):
        super(VGG16, self).__init__(vgg_name='VGG16', num_channels=num_channels, num_classes=num_classes)


class VGG16_bn(VGG):
    def __init__(self, num_channels=3, num_classes=10):
        super(VGG16_bn, self).__init__(vgg_name='VGG16', num_channels=num_channels, num_classes=num_classes,
                                       batch_norm=True)


class VGG19(VGG):
    def __init__(self, num_channels=3, num_classes=10):
        super(VGG19, self).__init__(vgg_name='VGG19', num_channels=num_channels, num_classes=num_classes)


class VGG19_bn(VGG):
    def __init__(self, num_channels=3, num_classes=10):
        super(VGG19_bn, self).__init__(vgg_name='VGG19', num_channels=num_channels, num_classes=num_classes,
                                       batch_norm=True)
