from src.image_classification.experiment import configuration, run

config = configuration()  # configuration of the experiments
savdir = f'{config["basepath"]}/results/image-classification/eval/'

for dataset in config['datasets']:
    savpath = f'{savdir}{dataset}-3.pickle'
    run(
        init=0,
        device=config['device'],
        dataset=dataset,
        models=config['models'],
        unfreezed_list=config['unfreezed'],
        learning_rates=config['learning_rates'],
        batch_sizes=config['batch_sizes'],
        K=config['K'],
        iterations_list=config['iterations'],
        experiments=config['experiments'],
        savpath=savpath
    )
