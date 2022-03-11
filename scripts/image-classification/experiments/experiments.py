from src.image_classification.experiment import run, configuration

config = configuration()  # configuration of the experiments
savdir = f'{config["basepath"]}/results/image-classification/eval/'

for dataset in config['datasets']:
    savpath = f'{savdir}{dataset}.pickle'
    run(
        config['device'],
        dataset,
        config['models'],
        config['unfreezed'],
        config['learning_rates'],
        config['batch_sizes'],
        config['K'],
        config['iterations'],
        config['experiments'],
        savpath
    )
