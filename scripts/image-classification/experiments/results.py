import pickle
from src.image_classification.experiment import configuration, plot_accuracy, plot_time, get_best_configuration_per_k

config = configuration()
resdir = f'{config["basepath"]}/results/image-classification/eval/'
savdir = f'{config["basepath"]}/results/image-classification/figs/'
savfig = True
mode = 'efficient'  # best, efficient
threshold = 0.02

for dataset in config['datasets']:
    filename = f'{dataset}.pickle'
    path = f'{resdir}{filename}'

    with open(path, 'rb') as h:
        results = pickle.load(h)

    for k in config['K']:
        best, accuracy = get_best_configuration_per_k(results, k, mode, threshold, savfig, savdir)

    plot_accuracy(dataset, results, 'model', config['models'], savfig, savdir)
    plot_accuracy(dataset, results, 'learning_rate', config['learning_rates'], savfig, savdir)
    plot_accuracy(dataset, results, 'batch_size', config['batch_sizes'], savfig, savdir)
    plot_accuracy(dataset, results, 'k', config['K'], savfig, savdir)
    plot_accuracy(dataset, results, 'iterations', config['iterations'], savfig, savdir)

    plot_time(dataset, results, 'model', config['models'], savfig, savdir)
    plot_time(dataset, results, 'learning_rate', config['learning_rates'], savfig, savdir)
    plot_time(dataset, results, 'batch_size', config['batch_sizes'], savfig, savdir)
    plot_time(dataset, results, 'k', config['K'], savfig, savdir)
    plot_time(dataset, results, 'iterations', config['iterations'], savfig, savdir)
