import os
import numpy as np
import pickle
import time
import timm
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms


def load_train_data(dataset, model, batch_size, k):
    # mv-model-building-gui absolute path
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    precrop = 256 if 'vit' in model else 160
    crop = 224 if 'vit' in model else 128
    if dataset in ['mnist', 'fashion-mnist']:
        transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize(precrop),
                transforms.RandomCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(precrop),
                transforms.RandomCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )

    if dataset == 'cifar10':
        train = torchvision.datasets.CIFAR10(
            root=f'{basepath}/data/image-classification',
            train=True,
            download=False,
            transform=transform
        )
    elif dataset == 'mnist':
        train = torchvision.datasets.MNIST(
            root=f'{basepath}/data/image-classification',
            train=True,
            download=False,
            transform=transform
        )
    elif dataset == 'fashion-mnist':
        train = torchvision.datasets.FashionMNIST(
            root=f'{basepath}/data/image-classification',
            train=True,
            download=False,
            transform=transform
        )
    else:
        raise Exception(f'{dataset} dataset is not implemented.')

    loader = torch.utils.data.DataLoader(
        train,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    train = []
    class_size = [0] * 10
    for data in loader:
        inputs, labels = data
        for c in range(10):
            if labels[0] == c and class_size[c] < k:
                train.append([inputs.squeeze(), labels.squeeze()])
                class_size[c] += 1
        if all([s == k for s in class_size]):
            break

    if batch_size <= len(train):
        loader = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
    else:
        loader = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            sampler=torch.utils.data.RandomSampler(
                train,
                replacement=True,
                num_samples=batch_size
            )
        )

    return loader


def load_test_data(dataset, model, batch_size):
    # mv-model-building-gui absolute path
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    size = 224 if 'vit' in model else 160
    if dataset in ['mnist', 'fashion-mnist']:
        transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )

    if dataset == 'cifar10':
        test = torchvision.datasets.CIFAR10(
            root=f'{basepath}/data/image-classification',
            train=False,
            download=False,
            transform=transform
        )
    elif dataset == 'mnist':
        test = torchvision.datasets.MNIST(
            root=f'{basepath}/data/image-classification',
            train=False,
            download=False,
            transform=transform
        )
    elif dataset == 'fashion-mnist':
        test = torchvision.datasets.FashionMNIST(
            root=f'{basepath}/data/image-classification',
            train=False,
            download=False,
            transform=transform
        )
    else:
        raise Exception(f'{dataset} dataset is not implemented.')

    loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return loader


def freeze_conv(network, unfreezed):
    layers = [name
              for name, param
              in network.named_parameters()
              if 'conv' in name]
    if unfreezed == 0.0:
        unfreezed_layers = []
    elif unfreezed == 'all':
        unfreezed_layers = layers
    else:
        unfreezed_layers = layers[-int(unfreezed * len(layers)):]

    for name, param in network.named_parameters():
        if 'fc' in name or name in unfreezed_layers:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return network


def freeze_blocks(network, unfreezed):
    layers = [name
              for name, param
              in network.named_parameters()
              if 'blocks' in name]
    if unfreezed == 0.0:
        unfreezed_layers = []
    elif unfreezed == 'all':
        unfreezed_layers = layers
    else:
        unfreezed_layers = layers[-int(unfreezed * len(layers)):]

    for name, param in network.named_parameters():
        if 'head' in name or name in unfreezed_layers:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return network


def train(network, device, loader, iterations, optimizer, learning_rate, criterion, showtxt):
    iteration = 0
    flag = False
    while True:
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iteration += 1

            if showtxt:
                print(f'[{iteration}] loss: {loss.item():.3f}')

            if iteration in [int(iterations * 0.3),
                             int(iterations * 0.6),
                             int(iterations * 0.9)]:
                learning_rate /= 10
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate

            if iteration == iterations:
                flag = True
                break

        if flag:
            break

    return network


def evaluate(network, device, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def run(
        device,
        dataset,
        savpath,
        models,
        unfreezed_list,
        learning_rates,
        batch_sizes,
        K,
        iterations_list,
        experiments):

    start_experiments = time.time()

    # Total number of experiments to be conducted
    total = len(models) * len(unfreezed_list) * len(learning_rates) * len(batch_sizes) * len(K) * len(iterations_list) * experiments
    results = []
    for model in models:
        test_loader = load_test_data(dataset, model, 128)
        for unfreezed in unfreezed_list:
            for learning_rate in learning_rates:
                for batch_size in batch_sizes:
                    for k in K:
                        for iterations in iterations_list:
                            accuracies = []
                            times = []
                            for experiment in range(experiments):
                                start = time.time()

                                print(f'{experiment + 1}..', end='')

                                train_loader = load_train_data(dataset, model, batch_size, k)
                                for data in train_loader:
                                    inputs, labels = data

                                network = timm.create_model(
                                    model,
                                    pretrained=True,
                                    num_classes=10
                                ).to(device)
                                if 'vit' in model:
                                    network = freeze_blocks(network, unfreezed)
                                else:
                                    network = freeze_conv(network, unfreezed)
                                criterion = torch.nn.CrossEntropyLoss()
                                optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

                                network = train(
                                    network=network,
                                    device=device,
                                    loader=train_loader,
                                    iterations=iterations,
                                    optimizer=optimizer,
                                    learning_rate=learning_rate,
                                    criterion=criterion,
                                    showtxt=False
                                )

                                accuracy = evaluate(
                                    network=network,
                                    device=device,
                                    loader=test_loader
                                )

                                accuracies.append(accuracy)

                                end = time.time()

                                times.append(end - start)

                            result = {
                                'config': {
                                    'model': model,
                                    'unfreezed': unfreezed,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'k': k,
                                    'iterations': iterations
                                },
                                'accuracy': accuracies
                            }
                            results.append(result)

                            with open(savpath, 'wb') as h:
                                pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

                            print(
                                f'\n[{dataset} - {len(results) * experiments}/{total}] ETA: {np.mean(times) * (total - len(results)) / 3600:1.2f} h')
                            print(result)
                            print()

    print(f'Total time: {(time.time() - start_experiments) / 3600:1.2f} h')
