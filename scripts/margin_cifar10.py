import numpy as np
import torch
import torch.nn as nn
import os
import time

from utils import get_dataset_loaders
from utils import train
from utils import generate_subspace_list
from utils import compute_margin_distribution
from model_classes import TransformLayer
from model_classes.cifar10 import ResNet18  # check inside the model_class.cifar10 package for other network options


TREE_ROOT = './'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET = 'CIFAR10'
PRETRAINED = True
PRETRAINED_PATH = '../Models/Pretrained/CIFAR10/ResNet18/model.t7'
BATCH_SIZE = 128


#############################
# Dataset paths and loaders #
#############################

# Specify the path of the dataset. For MNIST and CIFAR-10 the train and validation paths can be the same.
# For ImageNet, please specify to proper train and validation paths.
DATASET_DIR = {'train': os.path.join(TREE_ROOT, '../Datasets/CIFAR10/'),
               'val': os.path.join(TREE_ROOT, '../Datasets/CIFAR10/')
               }
os.makedirs(DATASET_DIR['train'], exist_ok=True)
os.makedirs(DATASET_DIR['val'], exist_ok=True)

# Load the data
trainloader, testloader, trainset, testset, mean, std = get_dataset_loaders(DATASET, DATASET_DIR, BATCH_SIZE)


####################
# Select a Network #
####################

# Normalization layer
trans = TransformLayer(mean=mean, std=std)

# Load a model
model = ResNet18()  # check inside the model_class.cifar10 package for other network options

# If pretrained
if PRETRAINED:
    print('---> Working on a pretrained network')
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location='cpu'))
    model = model.to(DEVICE)
    model.eval()

# If not pretrained, then train it
if not PRETRAINED:

    EPOCHS = 30
    MAX_LR = 0.21
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    opt = torch.optim.SGD(model.parameters(), lr=MAX_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    loss_fun = nn.CrossEntropyLoss()
    lr_schedule = lambda t: np.interp([t], [0, EPOCHS * 2 // 5, EPOCHS], [0, MAX_LR, 0])[0]  # Triangular (cyclic) learning rate schedule

    SAVE_TRAIN_DIR = os.path.join(TREE_ROOT, '../Models/Generated/%s/%s/' % (DATASET, model.__class__.__name__))
    os.makedirs(SAVE_TRAIN_DIR, exist_ok=True)

    t0 = time.time()
    model = model.to(DEVICE)
    model = train(model, trans, trainloader, testloader, EPOCHS, opt, loss_fun, lr_schedule, SAVE_TRAIN_DIR)

    print('---> Training is done! Elapsed time: %.5f minutes\n' % ((time.time() - t0) / 60.))


##################################
# Compute margin along subspaces #
##################################

# Create a list of subspaces to evaluate the margin on
SUBSPACE_DIM = 8
DIM = 32
SUBSPACE_STEP = 2

subspace_list = generate_subspace_list(SUBSPACE_DIM, DIM, SUBSPACE_STEP, channels=3)

# Select the data samples for evaluation
NUM_SAMPLES_EVAL = 100
indices = np.random.choice(len(testset), NUM_SAMPLES_EVAL, replace=False)

eval_dataset = torch.utils.data.Subset(testset, indices[:NUM_SAMPLES_EVAL])
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2, pin_memory=True if DEVICE == 'cuda' else False)

# Compute the margin using subspace DeepFool and save the results
RESULTS_DIR = os.path.join(TREE_ROOT, '../Results/margin_%s/%s/' % (DATASET, model.__class__.__name__))
os.makedirs(RESULTS_DIR, exist_ok=True)

margins = compute_margin_distribution(model, trans, eval_loader, subspace_list, RESULTS_DIR + 'margins.npy')
