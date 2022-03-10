import pickle
import json
import numpy as np
from src.image_classification.experiment import plt, plot_accuracy

drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
directory = f'{drive}PycharmProjects/mv-model-building-gui/results/image-classification/eval/'
savdir = f'{drive}PycharmProjects/mv-model-building-gui/results/image-classification/figs/'
savfig = True

# Hyper-parameter grid
models = [
    'resnet18',
    'vit_small_patch16_224_in21k',
    'resnetv2_50x3_bitm_in21k',
]
unfreezed_list = [
    0.0,
]
learning_rates = [
    0.01,
    0.003,
]
batch_sizes = [
    32,
    128,
]
K = [
    5,
    10,
    50,
]
iterations_list = [
    500,
    1000,
]
datasets = [
    'cifar10',
    'mnist',
    'fashion-mnist',
    # '400-bird-species'
]

for dataset in datasets:
    filename = f'{dataset}.pickle'
    path = f'{directory}{filename}'

    with open(path, 'rb') as h:
        results = pickle.load(h)

    for k in K:
        max_mean_accuracy = np.max([np.mean(r['accuracy']) for r in results if r['config']['k'] == k])
        best = [r for r in results if r['config']['k'] == k and
                np.mean(r['accuracy']) == max_mean_accuracy][0]

        print(f'[mean] k={k}, accuracy={max_mean_accuracy * 100:1.2f}%')
        print(json.dumps(best, sort_keys=False, indent=4))

        print()

    plot_accuracy(dataset, results, 'model', models, savfig, savdir)
    plot_accuracy(dataset, results, 'learning_rate', learning_rates, savfig, savdir)
    plot_accuracy(dataset, results, 'batch_size', batch_sizes, savfig, savdir)
    plot_accuracy(dataset, results, 'k', K, savfig, savdir)
    plot_accuracy(dataset, results, 'iterations', iterations_list, savfig, savdir)
