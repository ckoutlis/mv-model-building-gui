import pickle
import matplotlib.pyplot as plt
from src.image_classification.experiment import configuration, plot_results, get_best_configuration_per_k, plot_time_vs_accuracy

config = configuration()
resdir = f'{config["basepath"]}/results/image-classification/eval/'
savdir = f'{config["basepath"]}/results/image-classification/figs/'
savfig = True
mode = 'efficient'  # best, efficient
threshold = 0.05

fig1, axs1 = plt.subplots(4, 5, figsize=(12, 8))
fig2, axs2 = plt.subplots(4, 3, figsize=(7.2, 8))

for i, dataset in enumerate(config['datasets']):
    with open(f'{resdir}{dataset}.pickle', 'rb') as h:
        results = pickle.load(h)

    for j, k in enumerate(config['K']):
        times, accuracies, configs, duration, accuracy, best = get_best_configuration_per_k(results, k, mode, threshold)
        plot_time_vs_accuracy(
            dataset,
            k,
            times,
            accuracies,
            threshold,
            duration,
            accuracy,
            fig2,
            axs2[i, j],
            i == 0,
            i == len(config['datasets']) - 1,
            j == 0,
            i == 0 and j == 0
        )

    plot_results(dataset, results, 'model', config['models'], fig1, axs1[i, 0], x_bool=i == len(config['datasets']) - 1, y1_bool=True, y2_bool=False)
    plot_results(dataset, results, 'k', config['K'], fig1, axs1[i, 1], x_bool=i == len(config['datasets']) - 1, y1_bool=False, y2_bool=False)
    plot_results(dataset, results, 'iterations', config['iterations'], fig1, axs1[i, 2], x_bool=i == len(config['datasets']) - 1, y1_bool=False, y2_bool=False)
    plot_results(dataset, results, 'learning_rate', config['learning_rates'], fig1, axs1[i, 3], x_bool=i == len(config['datasets']) - 1, y1_bool=False, y2_bool=False)
    plot_results(dataset, results, 'batch_size', config['batch_sizes'], fig1, axs1[i, 4], x_bool=i == len(config['datasets']) - 1, y1_bool=False, y2_bool=True)

if savfig:
    fig1.savefig(f'{savdir}results.jpg')
    if mode == 'efficient':
        fig2.savefig(f'{savdir}time-vs-accuracy.jpg')
