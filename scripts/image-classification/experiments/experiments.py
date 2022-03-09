from src.image_classification.experiment import run
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
experiments = 3
datasets = [
    'cifar10',
    'mnist',
    'fashion-mnist'
]
# save results path
drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
savdir = f'{drive}PycharmProjects/mv-model-building-gui/results/image-classification/'

for dataset in datasets:
    savpath = f'{savdir}{dataset}.pickle'
    run(
        device,
        dataset,
        savpath,
        models,
        unfreezed_list,
        learning_rates,
        batch_sizes,
        K,
        iterations_list,
        experiments
    )
