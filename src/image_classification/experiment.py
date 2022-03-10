import os
import json
import random
import numpy as np
import pickle
import time
import timm
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


def load_train_data(dataset, model, batch_size, k):
    # mv-model-building-gui absolute path
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # resize = 256 if 'vit' in model else 160
    # crop = 224 if 'vit' in model else 128
    params = transform_params('train', dataset, model)

    if dataset in ['mnist', 'fashion-mnist']:
        transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize(params['resize']),
                transforms.RandomCrop(params['crop']),
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
                transforms.Resize(params['resize']),
                transforms.RandomCrop(params['crop']),
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
    elif dataset == '400-bird-species':
        drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
        datdir = f'{drive}DataStorage/400-bird-species'
        train = ImageFolder(f'{datdir}/train', transform=transform)
    else:
        raise Exception(f'{dataset} dataset is not implemented.')

    num_classes = len(train.classes)
    loader = torch.utils.data.DataLoader(
        train,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    train = []
    class_size = [0] * num_classes
    for data in loader:
        inputs, labels = data
        for c in range(num_classes):
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

    return loader, num_classes


def load_test_data(dataset, model, batch_size):
    # mv-model-building-gui absolute path
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # resize = 224 if 'vit' in model else 160
    params = transform_params('test', dataset, model)

    if dataset in ['mnist', 'fashion-mnist']:
        transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize(params['resize']),
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
                transforms.Resize(params['resize']),
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
    elif dataset == '400-bird-species':
        drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
        datdir = f'{drive}DataStorage/400-bird-species'
        test = ImageFolder(f'{datdir}/test', transform=transform)
    else:
        raise Exception(f'{dataset} dataset is not implemented.')

    loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return loader


def transform_params(mode, dataset, model):
    params = {
        'train': {
            'cifar10': {
                'resnet18': {
                    'resize': 160,
                    'crop': 128
                },
                'vit_small_patch16_224_in21k': {
                    'resize': 256,
                    'crop': 224
                },
                'resnetv2_50x3_bitm_in21k': {
                    'resize': 160,
                    'crop': 128
                }
            },
            'mnist': {
                'resnet18': {
                    'resize': 160,
                    'crop': 128
                },
                'vit_small_patch16_224_in21k': {
                    'resize': 256,
                    'crop': 224
                },
                'resnetv2_50x3_bitm_in21k': {
                    'resize': 160,
                    'crop': 128
                }
            },
            'fashion-mnist': {
                'resnet18': {
                    'resize': 160,
                    'crop': 128
                },
                'vit_small_patch16_224_in21k': {
                    'resize': 256,
                    'crop': 224
                },
                'resnetv2_50x3_bitm_in21k': {
                    'resize': 160,
                    'crop': 128
                }
            },
            '400-bird-species': {
                'resnet18': {
                    'resize': 256,
                    'crop': 224
                },
                'vit_small_patch16_224_in21k': {
                    'resize': 256,
                    'crop': 224
                },
                'resnetv2_50x3_bitm_in21k': {
                    'resize': 256,
                    'crop': 224
                }
            }
        },
        'test': {
            'cifar10': {
                'resnet18': {
                    'resize': 160,
                },
                'vit_small_patch16_224_in21k': {
                    'resize': 224,
                },
                'resnetv2_50x3_bitm_in21k': {
                    'resize': 160,
                }
            },
            'mnist': {
                'resnet18': {
                    'resize': 160,
                },
                'vit_small_patch16_224_in21k': {
                    'resize': 224,
                },
                'resnetv2_50x3_bitm_in21k': {
                    'resize': 160,
                }
            },
            'fashion-mnist': {
                'resnet18': {
                    'resize': 160,
                },
                'vit_small_patch16_224_in21k': {
                    'resize': 224,
                },
                'resnetv2_50x3_bitm_in21k': {
                    'resize': 160,
                }
            },
            '400-bird-species': {
                'resnet18': {
                    'resize': 224,
                },
                'vit_small_patch16_224_in21k': {
                    'resize': 224,
                },
                'resnetv2_50x3_bitm_in21k': {
                    'resize': 224,
                }
            }
        }
    }

    return params[mode][dataset][model]


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
                            seeds = []
                            for experiment in range(experiments):
                                start = time.time()

                                # Make the experiments reproducible
                                seed = len(results) * experiments + experiment
                                seeds.append(seed)
                                torch.manual_seed(seed)
                                random.seed(seed)
                                np.random.seed(seed)

                                print(f'{experiment + 1}..', end='')

                                train_loader, num_classes = load_train_data(dataset, model, batch_size, k)

                                network = timm.create_model(
                                    model,
                                    pretrained=True,
                                    num_classes=num_classes
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
                                'dataset': dataset,
                                'config': {
                                    'k': k,
                                    'model': model,
                                    'unfreezed': unfreezed,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'iterations': iterations
                                },
                                'accuracy': accuracies,
                                'time': times,
                                'seed': seeds,
                                'total': total
                            }
                            results.append(result)

                            with open(savpath, 'wb') as h:
                                pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

                            print(f'\n[{dataset} - {len(results) * experiments}/{total}] '
                                  f'ETA: {np.mean(times) * (total - len(results) * experiments) / 3600:1.2f} h')
                            print(json.dumps(result, sort_keys=False, indent=4))
                            print()

    print(f'Total time: {(time.time() - start_experiments) / 3600:1.2f} h\n')


def plot_accuracy(dataset, results, hyperparameter, values, savfig, savdir):
    accuracy_max = []
    accuracy_all = []
    for value in values:
        accuracy_max.append(
            np.max([np.mean(r['accuracy'])
                    for r in results
                    if r['config'][hyperparameter] == value])
        )
        accuracy_all.append(
            [a for r in results for a in r['accuracy'] if r['config'][hyperparameter] == value]
        )
    plt.figure(figsize=(9, 5))
    plt.suptitle(f'{dataset} - {hyperparameter}')
    plt.subplot(1, 2, 1)
    plt.title('max accuracy')
    plt.bar([str(x) for x in values], accuracy_max)
    plt.xticks(rotation=8)
    plt.ylim([max(np.min(accuracy_max) - 0.2, 0), 1.0])
    plt.subplot(1, 2, 2)
    plt.title('distribution of accuracy')
    plt.boxplot(accuracy_all)
    plt.xticks(np.arange(1, len(values) + 1), values, rotation=8)

    if savfig:
        plt.savefig(f'{savdir}accuracy-{dataset}-{hyperparameter}.jpg')
