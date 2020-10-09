import torch
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class MnistResNet(ResNet):
    def __init__(self, block, layers, num_classes=10):
        super(MnistResNet, self).__init__(block=block, layers=layers, num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return super(MnistResNet, self).forward(x)


def ResNet18(num_classes=10):
    return MnistResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return MnistResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return MnistResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return MnistResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return MnistResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
