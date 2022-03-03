import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def load_train_data(model, batch_size, k):
    # mv-model-building-gui absolute path
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    autoaugment = transforms.AutoAugment()
    totensor = transforms.ToTensor()
    resize = transforms.Resize(224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if 'vit' in model:
        transform = transforms.Compose([autoaugment, totensor, resize, normalize])
    else:
        transform = transforms.Compose([autoaugment, totensor, normalize])

    trainset = torchvision.datasets.CIFAR10(root=f'{basepath}/data/image-classification', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

    traindata = []
    class_size = [0] * 10
    for data in trainloader:
        inputs, labels = data

        for c in range(10):
            if labels[0] == c and class_size[c] < k:
                traindata.append([inputs, labels])
                class_size[c] += 1

        if all([s == k for s in class_size]):
            break

    batched_train_data = []
    for i in range(np.ceil(len(traindata) / batch_size).astype(int)):
        batch_image = []
        batch_label = []
        for data in traindata[i * batch_size:(i + 1) * batch_size]:
            batch_image.append(data[0])
            batch_label.append(data[1])
        batched_train_data.append([torch.cat(batch_image, 0), torch.cat(batch_label, 0)])
    return batched_train_data


def load_test_data(model, batch_size):
    # mv-model-building-gui absolute path
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    autoaugment = transforms.AutoAugment()
    totensor = transforms.ToTensor()
    resize = transforms.Resize(224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if 'vit' in model:
        transform = transforms.Compose([autoaugment, totensor, resize, normalize])
    else:
        transform = transforms.Compose([autoaugment, totensor, normalize])

    testset = torchvision.datasets.CIFAR10(root=f'{basepath}/data/image-classification', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader


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


def train(network, device, loader, iterations, optimizer, criterion, showtxt):
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
