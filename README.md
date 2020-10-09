# Hold me tight! Influence of discriminative features on deep network boundaries
This is the source code to reproduce the experiments of the NeurIPS 2020 paper "[Hold me tight! Influence of discriminative features on deep network boundaries](https://arxiv.org/abs/2002.06349)" by Guillermo Ortiz-Jimenez*, Apostolos Modas*, Seyed-Mohsen Moosavi-Dezfooli and Pascal Frossard.

## Abstract
Important insights towards the explainability of neural networks reside in the characteristics of their decision boundaries. In this work, we borrow tools from the field of adversarial robustness, and propose a new perspective that relates dataset features to the distance of samples to the decision boundary. This enables us to carefully tweak the position of the training samples and measure the induced changes on the boundaries of CNNs trained on large-scale vision datasets. We use this framework to reveal some intriguing properties of CNNs. Specifically, we rigorously confirm that neural networks exhibit a high invariance to non-discriminative features, and show that very small perturbations of the training samples in certain directions can lead to sudden invariances in the orthogonal ones. This is precisely the mechanism that adversarial training uses to achieve robustness.

## Dependencies
To run our code on a Linux machine with a GPU, install the Python packages in a fresh Anaconda environment:
```
$ conda env create -f environment.yml
$ conda activate hold_me_tight
```

Note: this repository comes with some pretrained [models](Models/Pretrained/), which have been uploaded as Git LFS. In case you do not want to clone the pretrained models, i.e., due to disk space limitations, please clone the repository using the following command:
```
$ GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:LTS4/hold-me-tight.git
```

## Experiments
This repository contains code to reproduce the following experiments: 

- Train and compute the margin distribution on [MNIST](scripts/margin_mnist.py)
- Train and compute the margin distribution on [Frequency-flipped MNIST](scripts/margin_flipped_mnist.py)
- Train and compute the margin distribution on [CIFAR10](scripts/margin_cifar10.py)
- Train and compute the margin distribution on [Frequency-flipped CIFAR10](scripts/margin_flipped_cifar10.py)
- Train and compute the margin distribution on [Low-Pass CIFAR10](scripts/margin_low_pass_cifar10.py)
- Compute the margin distribution on [Robust MNIST](scripts/margin_robust_mnist.py)
- Compute the margin distribution on [Robust CIFAR10](scripts/margin_robust_cifar10.py)
- Compute the margin distribution on [Robust ImageNet](scripts/margin_robust_imagenet.py)
- Compute the margin distribution on [Frequency-flipped ImageNet](scripts/margin_flipped_imagenet.py)

You can reproduce this experiments separately using their individual scripts, or have a look at the comprehensive [Jupyter notebook](Hold_Me_Tight.ipynb).

## Pretrained architectures

The repository also contains a set of pretrained models that we used in our experiments. The exact hyperparameters and settings can be found in the Supplementary material of the [paper](https://arxiv.org/abs/2002.06349).

Architecture | Dataset | Training method
---|---|---
[LeNet](Models/Pretrained/MNIST/LeNet/) | MNIST | Standard
[ResNet18](Models/Pretrained/MNIST/ResNet18/) | MNIST | Standard
[ResNet18](Models/Pretrained/CIFAR10/ResNet18/) | CIFAR10 | Standard
[VGG19](Models/Pretrained/CIFAR10/VGG19/) | CIFAR10 | Standard
[DenseNet121](Models/Pretrained/CIFAR10/DenseNet121/) | CIFAR10 | Standard
[LeNet](Models/Pretrained/MNIST_flipped/LeNet/) | Flipped MNIST | Standard + Frequency flip
[ResNet18](Models/Pretrained/MNIST_flipped/ResNet18/) | Flipped MNIST | Standard + Frequency flip
[ResNet18](Models/Pretrained/CIFAR10_flipped/ResNet18/) | Flipped CIFAR10 | Standard + Frequency flip
[VGG19](Models/Pretrained/CIFAR10_flipped/VGG19/) | Flipped CIFAR10 | Standard + Frequency flip
[DenseNet121](Models/Pretrained/CIFAR10_flipped/DenseNet121/) | Flipped CIFAR10 | Standard + Frequency flip
[ResNet50](Models/Pretrained/ImageNet_flipped/ResNet50/) | Flipped ImageNet | Standard + Frequency flip
[ResNet18](Models/Pretrained/CIFAR10_low_pass/ResNet18/) | Low-pass CIFAR10 | Standard + Low-pass filtering
[VGG19](Models/Pretrained/CIFAR10_low_pass/VGG19/) | Low-pass CIFAR10 | Standard + Low-pass filtering
[DenseNet121](Models/Pretrained/CIFAR10_low_pass/DenseNet121/) | Low-pass CIFAR10 | Standard + Low-pass filtering
[Robust LeNet](Models/Pretrained/MNIST_robust/LeNet/) | MNIST | L2 PGD adversarial training (eps = 2)
[Robust ResNet18](Models/Pretrained/MNIST_robust/ResNet18/) | MNIST | L2 PGD adversarial training (eps = 2)
[Robust ResNet18](Models/Pretrained/CIFAR10_robust/ResNet18/) | CIFAR10 | L2 PGD adversarial training (eps = 1)
[Robust VGG19](Models/Pretrained/CIFAR10_robust/VGG19/) | CIFAR10 | L2 PGD adversarial training (eps = 1)
[Robust DenseNet121](Models/Pretrained/CIFAR10_robust/DenseNet121/) | CIFAR10 | L2 PGD adversarial training (eps = 1)
[Robust ResNet50](Models/Pretrained/ImageNet_robust/ResNet50/) | ImageNet | L2 PGD adversarial training (eps = 3) (copied from [here](https://github.com/MadryLab/robustness))
[Robust LeNet](Models/Pretrained/MNIST_flipped_robust/LeNet/) | Flipped MNIST | L2 PGD adversarial training (eps = 2) with Dykstra projection + Frequency flip
[Robust ResNet18](Models/Pretrained/MNIST_flipped_robust/ResNet18/) | Flipped MNIST | L2 PGD adversarial training (eps = 2) with Dykstra projection + Frequency flip
[Robust ResNet18](Models/Pretrained/CIFAR10_flipped_robust/ResNet18/) | Flipped CIFAR10 | L2 PGD adversarial training (eps = 1) with Dykstra projection + Frequency flip
[Robust VGG19](Models/Pretrained/CIFAR10_flipped_robust/VGG19/) | Flipped CIFAR10 | L2 PGD adversarial training (eps = 1) with Dykstra projection + Frequency flip
[Robust DenseNet121](Models/Pretrained/CIFAR10_flipped_robust/DenseNet121/) | Flipped CIFAR10 | L2 PGD adversarial training (eps = 1) with Dykstra projection + Frequency flip

## Reference
If you use this code, or some of the attached models, please cite the following [paper](https://arxiv.org/abs/2002.06349):

```
@InCollection{OrtizModasNADs2020,
  TITLE = {{Hold me tight! Influence of discriminative features on deep network boundaries}},
  AUTHOR = {{Ortiz-Jimenez}, Guillermo and {Modas}, Apostolos and {Moosavi-Dezfooli}, Seyed-Mohsen and Frossard, Pascal},
  BOOKTITLE = {Advances in Neural Information Processing Systems 34},
  MONTH = dec,
  YEAR = {2020}
}
```
