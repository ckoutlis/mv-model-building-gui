from src.image_classification.experiment import run
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper-parameter grid
models = [
    'resnet18',
    # 'vit_small_patch16_224_in21k',
    # 'resnetv2_50x1_bitm_in21k',
    # 'resnetv2_101x3_bitm_in21k'
]
unfreezed_list = [
    0.0,
    # 0.3,
    # 'all'
]
learning_rates = [
    # 0.001,
    0.003,
]
batch_sizes = [
    # 8,
    # 16,
    # 32,
    # 64,
    # 128,
    # 256,
    512
]
K = [
    5,
    # 10,
    # 50,
]
iterations_list = [
    500,
    # 1000,
    # 2000,
    # 3000
]
experiments = 1

# save results path
datasets = ['cifar10']
drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
savdir = f'{drive}PycharmProjects/mv-model-building-gui/results/image-classification/'

for dataset in datasets:
    savpath = f'{savdir}{dataset}_accuracy.pickle'
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
