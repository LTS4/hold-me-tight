## Pretrained architectures

This directory provides a set of pretrained models that we used in our experiments. The exact hyperparameters and settings can be found in the Supplementary material of the [paper](https://arxiv.org/abs/2002.06349). All the models are publicly available and can be downloaded from [here](https://drive.google.com/drive/folders/1Y4SWnfdojzODX2SQyf8V5kC6sE1pmF4m?usp=sharing). In order to execute the scripts using the pretrained models, it is recommended to download and save them undert this directory. 

Architecture | Dataset | Training method
---|---|---
LeNet | MNIST | Standard
ResNet18 | MNIST | Standard
ResNet18 | CIFAR10 | Standard
VGG19 | CIFAR10 | Standard
DenseNet121 | CIFAR10 | Standard
LeNet | Flipped MNIST | Standard + Frequency flip
ResNet18 | Flipped MNIST | Standard + Frequency flip
ResNet18 | Flipped CIFAR10 | Standard + Frequency flip
VGG19 | Flipped CIFAR10 | Standard + Frequency flip
DenseNet121 | Flipped CIFAR10 | Standard + Frequency flip
ResNet50 | Flipped ImageNet | Standard + Frequency flip
ResNet18 | Low-pass CIFAR10 | Standard + Low-pass filtering
VGG19 | Low-pass CIFAR10 | Standard + Low-pass filtering
DenseNet121 | Low-pass CIFAR10 | Standard + Low-pass filtering
Robust LeNet | MNIST | L2 PGD adversarial training (eps = 2)
Robust ResNet18 | MNIST | L2 PGD adversarial training (eps = 2)
Robust ResNet18 | CIFAR10 | L2 PGD adversarial training (eps = 1)
Robust VGG19 | CIFAR10 | L2 PGD adversarial training (eps = 1)
Robust DenseNet121 | CIFAR10 | L2 PGD adversarial training (eps = 1)
Robust ResNet50 | ImageNet | L2 PGD adversarial training (eps = 3) (copied from [here](https://github.com/MadryLab/robustness))
Robust LeNet | Flipped MNIST | L2 PGD adversarial training (eps = 2) with Dykstra projection + Frequency flip
Robust ResNet18 | Flipped MNIST | L2 PGD adversarial training (eps = 2) with Dykstra projection + Frequency flip
Robust ResNet18 | Flipped CIFAR10 | L2 PGD adversarial training (eps = 1) with Dykstra projection + Frequency flip
Robust VGG19 | Flipped CIFAR10 | L2 PGD adversarial training (eps = 1) with Dykstra projection + Frequency flip
Robust DenseNet121 | Flipped CIFAR10 | L2 PGD adversarial training (eps = 1) with Dykstra projection + Frequency flip

## Reference
If you use some of the attached models, please cite the following [paper](https://arxiv.org/abs/2002.06349):

```
@InCollection{OrtizModasHMT2020,
  TITLE = {{Hold me tight! Influence of discriminative features on deep network boundaries}},
  AUTHOR = {{Ortiz-Jimenez}, Guillermo and {Modas}, Apostolos and {Moosavi-Dezfooli}, Seyed-Mohsen and Frossard, Pascal},
  BOOKTITLE = {Advances in Neural Information Processing Systems 34},
  MONTH = dec,
  YEAR = {2020}
}
```
