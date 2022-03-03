import pickle
import time
import timm
import torch.optim as optim
from src.image_classification.experiment import *

start_experiments = time.time()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# save results directory
drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
savdir = f'{drive}PycharmProjects/mv-model-building-gui/results/image-classification/'

# Hyper-parameter grid
models = [
    # 'resnet18',
    # 'vit_small_patch16_224_in21k',
    'resnetv2_101x3_bitm_in21k',
]
unfreezed_list = [
    0.,
    0.3,
    'all'
]
learning_rates = [
    0.0001,
    0.00001
]
batch_sizes = [
    8,
    32
]
K = [
    10,
    50,
]
iterations_list = [
    500,
    1000,
]
experiments = 3

# Total number of experiments to be conducted
total = len(models) * len(unfreezed_list) * len(learning_rates) * \
        len(batch_sizes) * len(K) * len(iterations_list) * experiments

results = []
for model in models:
    test_loader = load_test_data(model=model, batch_size=128)
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

                            train_loader = load_train_data(model=model, batch_size=batch_size, k=k)

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

                        with open(f'{savdir}cifar10_accuracy.pickle', 'wb') as h:
                            pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

                        print(f'\n[{len(results)}/{total}] ETA: {np.mean(times) * (total - len(results)) / 3600:1.2f} h')
                        print(result)
                        print()

print(f'Total time: {(time.time() - start_experiments) / 3600:1.2f} h')
