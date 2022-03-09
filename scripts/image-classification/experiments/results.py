import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
directory = f'{drive}PycharmProjects/mv-model-building-gui/results/image-classification/'
dataset = 'cifar10'  # fashion-mnist, mnist, cifar10
filename = f'{dataset}.pickle'
path = f'{directory}{filename}'

models = [
    'resnet18',
    'vit_small_patch16_224_in21k',
    'resnetv2_50x1_bitm_in21k',
    'resnetv2_50x3_bitm_in21k',
]
unfreezed_list = [
    0.0,
    # 0.3,
    # 'all'
]
learning_rates = [
    0.0005,
    0.001,
    0.003,
]
batch_sizes = [
    8,
    32,
    256,
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

with open(path, 'rb') as h:
    results = pickle.load(h)

for k in K:
    max_mean_accuracy = np.max([np.mean(r['accuracy']) for r in results if r['config']['k'] == k])
    best = [r for r in results if r['config']['k'] == k and
            np.mean(r['accuracy']) == max_mean_accuracy][0]

    print(f'[mean] k={k}, accuracy={max_mean_accuracy * 100:1.2f}%')
    print(json.dumps(best, sort_keys=False, indent=4))

    print()

accuracy = []
for model in models:
    accuracy.append(
        np.max([np.mean(r['accuracy'])
                for r in results
                if r['config']['model'] == model])
    )
plt.figure(figsize=(9, 4))
plt.bar(models, accuracy)
plt.ylim([max(np.min(accuracy) - 0.2, 0), 1.0])

accuracy = []
for learning_rate in learning_rates:
    accuracy.append(
        np.max([np.mean(r['accuracy'])
                for r in results
                if r['config']['learning_rate'] == learning_rate])
    )
plt.figure(figsize=(9, 4))
plt.bar([str(x) for x in learning_rates], accuracy)
plt.ylim([max(np.min(accuracy) - 0.2, 0), 1.0])

accuracy = []
for batch_size in batch_sizes:
    accuracy.append(
        np.max([np.mean(r['accuracy'])
                for r in results
                if r['config']['batch_size'] == batch_size])
    )
plt.figure(figsize=(9, 4))
plt.bar([str(x) for x in batch_sizes], accuracy)
plt.ylim([max(np.min(accuracy) - 0.2, 0), 1.0])

accuracy = []
for k in K:
    accuracy.append(
        np.max([np.mean(r['accuracy'])
                for r in results
                if r['config']['k'] == k])
    )
plt.figure(figsize=(9, 4))
plt.bar([str(x) for x in K], accuracy)
plt.ylim([max(np.min(accuracy) - 0.2, 0), 1.0])

accuracy = []
for iterations in iterations_list:
    accuracy.append(
        np.max([np.mean(r['accuracy'])
                for r in results
                if r['config']['iterations'] == iterations])
    )
plt.figure(figsize=(9, 4))
plt.bar([str(x) for x in iterations_list], accuracy)
plt.ylim([max(np.min(accuracy) - 0.2, 0), 1.0])

plt.show()
