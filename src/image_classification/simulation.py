import os
import json
import pickle
import timm
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch.utils.data


def configuration():
    config = {
        # mv-model-building-gui absolute path
        'basepath': os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'model': 'vit_small_patch16_224_in21k',
        'unfreezed': 0.0,
        'learning_rate': 0.01,
        'batch_size': 32,
        'iterations': 500,
    }
    return config


def load_train_data():
    datdir = f'{configuration()["basepath"]}/data/image-classification/simulation/train'
    batch_size = configuration()['batch_size']

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]
    )

    train = ImageFolder(f'{datdir}', transform=transform)

    if batch_size <= len(train):
        loader = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
    else:
        loader = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            sampler=torch.utils.data.RandomSampler(
                train,
                replacement=True,
                num_samples=batch_size
            )
        )

    return loader, train.classes


def load_model_to_train(num_classes):
    network = timm.create_model(
        configuration()['model'],
        pretrained=True,
        num_classes=num_classes
    ).to(configuration()['device'])
    network = freeze_blocks(network)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=configuration()['learning_rate'], momentum=0.9)
    return network, criterion, optimizer


def freeze_blocks(network):
    unfreezed = configuration()['unfreezed']

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


def train(network, loader, optimizer, criterion):
    device = configuration()['device']
    iterations = configuration()['iterations']
    learning_rate = configuration()['learning_rate']

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


def save_model(network, classes):
    basepath = configuration()['basepath']
    network_scripted = torch.jit.script(network)
    network_scripted.save(f'{basepath}/results/image-classification/models/classifier.pt')
    with open(f'{basepath}/results/image-classification/models/classes.pickle', 'wb') as h:
        pickle.dump(classes, h)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, datdir, transform):
        self.datdir = datdir
        self.transform = transform
        self.filepaths = os.listdir(self.datdir)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = os.path.join(self.datdir, self.filepaths[idx])
        image = Image.open(filepath).convert("RGB")
        tensor_image = self.transform(image)
        return self.filepaths[idx], tensor_image


def load_test_data():
    datdir = f'{configuration()["basepath"]}/data/image-classification/simulation/test'
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]
    )
    test = TestSet(datdir, transform)
    loader = torch.utils.data.DataLoader(
        test,
        batch_size=configuration()['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    return loader


def load_trained_model():
    model = torch.jit.load(f'{configuration()["basepath"]}/results/image-classification/models/classifier.pt')
    model.eval()
    return model


def infer(loader, model, showtxt):
    basepath = configuration()['basepath']
    with open(f'{basepath}/results/image-classification/models/classes.pickle', 'rb') as h:
        classes = pickle.load(h)

    device = configuration()['device']
    results = {}
    with torch.no_grad():
        for data in loader:
            filepaths, images = data[0], data[1].to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            probability, prediction = torch.max(probabilities.data, 1)
            for i, filepath in enumerate(filepaths):
                results[filepath] = {
                    'prediction': classes[prediction[i]],
                    'probability': float(probability[i].cpu())
                }
    if showtxt:
        print(json.dumps(results, sort_keys=False, indent=4))

    return results
