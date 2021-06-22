import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torchvision
import torchvision.transforms as transforms
import torch_dct
import numpy as np
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, trans, trainloader, testloader, epochs, opt, loss_fun, lr_schedule, save_train_dir):

    # lr_schedule = lambda t: np.interp([t], [0, epochs * 2 // 5, epochs], [0, max_lr, 0])[0]
    # loss_fun = nn.CrossEntropyLoss()

    print('Starting training...')
    print()

    for epoch in range(epochs):
        print('Epoch', epoch)
        train_loss_sum = 0
        train_acc_sum = 0
        train_n = 0

        model.train()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            opt.param_groups[0].update(lr=lr)

            output = model(trans(inputs))
            loss = loss_fun(output, targets)

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            train_loss_sum += loss.item() * targets.size(0)
            train_acc_sum += (output.max(1)[1] == targets).sum().item()
            train_n += targets.size(0)

            if batch_idx % 100 == 0:
                print('Batch idx: %d(%d)\tTrain Acc: %.3f%%\tTrain Loss: %.3f' %
                      (batch_idx, epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))

        print('\nTrain Summary\tEpoch: %d | Train Acc: %.3f%% | Train Loss: %.3f' %
              (epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))

        test_acc, test_loss = test(model, trans, testloader)
        print('Test  Summary\tEpoch: %d | Test Acc: %.3f%% | Test Loss: %.3f\n' % (epoch, test_acc, test_loss))

    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()

    torch.save(state_dict, save_train_dir + 'model.t7')

    return model


def test(model, trans, testloader):
    loss_fun = nn.CrossEntropyLoss()
    test_loss_sum = 0
    test_acc_sum = 0
    test_n = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            output = model(trans(inputs))
            loss = loss_fun(output, targets)

            test_loss_sum += loss.item() * targets.size(0)
            test_acc_sum += (output.max(1)[1] == targets).sum().item()
            test_n += targets.size(0)

        test_loss = (test_loss_sum / test_n)
        test_acc = (100 * test_acc_sum / test_n)

        return test_acc, test_loss


def subspace_deepfool(im, model, trans, num_classes=10, overshoot=0.02, max_iter=100, Sp=None, device=DEVICE):
    image = copy.deepcopy(im)
    input_shape = image.size()

    f_image = model(trans(Variable(image, requires_grad=True))).view((-1,))
    I = f_image.argsort(descending=True)
    I = I[0:num_classes]
    label_orig = I[0]

    pert_image = copy.deepcopy(image)

    r = torch.zeros(input_shape).to(device)

    label_pert = label_orig
    loop_i = 0

    while label_pert == label_orig and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = model(trans(x))

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            if Sp is None:
                pert_k = torch.abs(f_k) / w_k.norm()
            else:
                pert_k = torch.abs(f_k) / torch.matmul(Sp.t(), w_k.view([-1, 1])).norm()

            if pert_k < pert:
                pert = pert_k + 0.
                w = w_k + 0.

        if Sp is not None:
            w = torch.matmul(Sp, torch.matmul(Sp.t(), w.view([-1, 1]))).reshape(w.shape)

        r_i = torch.clamp(pert, min=1e-4) * w / w.norm()
        r = r + r_i

        pert_image = pert_image + r_i

        label_pert = torch.argmax(model(trans(Variable(image + (1 + overshoot) * r, requires_grad=False))).data).item()

        loop_i += 1

    return (1 + overshoot) * r, loop_i, label_orig, label_pert, image + (1 + overshoot) * r


def compute_margin_distribution(model, trans, dataloader, subspace_list, path, proc_fun=None):
    margins = []

    print('Measuring margin distribution...')
    for s, Sp in enumerate(subspace_list):
        Sp = Sp.to(DEVICE)
        sp_margin = []

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            if proc_fun:
                inputs = proc_fun(inputs)

            adv_perts = torch.zeros_like(inputs)
            for n, im in enumerate(inputs):
                adv_perts[n], _, _, _, _ = subspace_deepfool(im, model, trans, Sp=Sp)

            sp_margin.append(adv_perts.cpu().view([-1, np.prod(inputs.shape[1:])]).norm(dim=[1]))
        
        sp_margin = torch.cat(sp_margin)
        margins.append(sp_margin.numpy())
        print('Subspace %d:\tMedian margin: %5.5f' % (s, np.median(sp_margin)))

    np.save(path, margins)
    return np.array(margins)


def kron(a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def generate_subspace_list(subspace_dim, dim, subspace_step, channels):
    subspace_list = []
    idx_i = 0
    idx_j = 0
    while (idx_i + subspace_dim - 1 <= dim - 1) and (idx_j + subspace_dim - 1 <= dim - 1):

        S = torch.zeros((subspace_dim, subspace_dim, dim, dim), dtype=torch.float32).to(DEVICE)
        for i in range(subspace_dim):
            for j in range(subspace_dim):
                dirac = torch.zeros((dim, dim), dtype=torch.float32, device=DEVICE)
                dirac[idx_i + i, idx_j + j] = 1.
                S[i, j] = torch_dct.idct_2d(dirac, norm='ortho')

        Sp = S.view(subspace_dim * subspace_dim, dim * dim)
        if channels > 1:
            Sp = kron(torch.eye(channels, dtype=torch.float32, device=DEVICE), Sp)

        Sp = Sp.t()

        Sp = Sp.to('cpu')
        subspace_list.append(Sp)

        idx_i += subspace_step
        idx_j += subspace_step

    return subspace_list


def get_dataset_loaders(dataset, dataset_dir, batch_size=128):

    pin_memory = True if DEVICE == 'cuda' else False

    if dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root=dataset_dir['train'], download=True, train=True, transform=torchvision.transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=dataset_dir['val'], download=True, train=False, transform=torchvision.transforms.ToTensor())

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=pin_memory)

        mean = torch.tensor([0.1307], device=DEVICE)[None, :, None, None]
        std = torch.tensor([0.3081], device=DEVICE)[None, :, None, None]

    elif dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir['train'], download=True, train=True, transform=torchvision.transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=dataset_dir['val'], download=True, train=False, transform=torchvision.transforms.ToTensor())

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=pin_memory)

        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE)[None, :, None, None]
        std = torch.tensor([0.247, 0.243, 0.261], device=DEVICE)[None, :, None, None]

    elif dataset == 'ImageNet':

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.ImageFolder(root=dataset_dir['train'], transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=dataset_dir['val'], transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=4, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=4, pin_memory=True)

        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float, device=DEVICE)[None, :, None, None]
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float, device=DEVICE)[None, :, None, None]

    else:
        raise NotImplementedError

    return trainloader, testloader, trainset, testset, mean, std


def get_processed_dataset_loaders(proc_fun, dataset, dataset_dir, batch_size=128):

    pin_memory = True if DEVICE == 'cuda' else False

    if dataset == 'MNIST':
        orig_trainset = torchvision.datasets.MNIST(root=dataset_dir['train'], download=True, train=True, transform=torchvision.transforms.ToTensor())
        orig_testset = torchvision.datasets.MNIST(root=dataset_dir['val'], download=True, train=False, transform=torchvision.transforms.ToTensor())

        # trainset = torch.utils.data.TensorDataset(proc_fun(torch.tensor(trainset.data).type(torch.float32).permute([-1, 1, 28, 28]) / 255.), torch.tensor(trainset.targets))
        # testset = torch.utils.data.TensorDataset(proc_fun(torch.tensor(testset.data).type(torch.float32).permute([-1, 1, 28, 28]) / 255.), torch.tensor(testset.targets))

        trainset = torch.utils.data.TensorDataset(proc_fun(orig_trainset.data.type(torch.float32).unsqueeze(1) / 255.), orig_trainset.targets)
        testset = torch.utils.data.TensorDataset(proc_fun(orig_testset.data.type(torch.float32).unsqueeze(1) / 255.), orig_testset.targets)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=pin_memory)

        mean = torch.tensor([0.1307], device=DEVICE)[None, :, None, None]
        std = torch.tensor([0.3081], device=DEVICE)[None, :, None, None]

        proc_mean = torch.as_tensor(trainset.tensors[0].mean(axis=(0, 2, 3)), dtype=torch.float, device=DEVICE)[None, :, None, None]
        proc_std = torch.as_tensor(trainset.tensors[0].std(axis=(0, 2, 3)), dtype=torch.float, device=DEVICE)[None, :, None, None]

    elif dataset == 'CIFAR10':
        orig_trainset = torchvision.datasets.CIFAR10(root=dataset_dir['train'], download=True, train=True, transform=torchvision.transforms.ToTensor())
        orig_testset = torchvision.datasets.CIFAR10(root=dataset_dir['val'], download=True, train=False, transform=torchvision.transforms.ToTensor())

        trainset = torch.utils.data.TensorDataset(proc_fun(torch.tensor(orig_trainset.data).type(torch.float32).permute([0, 3, 1, 2]) / 255.), torch.tensor(orig_trainset.targets))
        testset = torch.utils.data.TensorDataset(proc_fun(torch.tensor(orig_testset.data).type(torch.float32).permute([0, 3, 1, 2]) / 255.), torch.tensor(orig_testset.targets))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=pin_memory)

        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE)[None, :, None, None]
        std = torch.tensor([0.247, 0.243, 0.261], device=DEVICE)[None, :, None, None]

        proc_mean = torch.as_tensor(trainset.tensors[0].mean(axis=(0, 2, 3)), dtype=torch.float, device=DEVICE)[None, :, None, None]
        proc_std = torch.as_tensor(trainset.tensors[0].std(axis=(0, 2, 3)), dtype=torch.float, device=DEVICE)[None, :, None, None]

    else:
        raise NotImplementedError

    return trainloader, testloader, trainset, testset, mean, std, proc_mean, proc_std
