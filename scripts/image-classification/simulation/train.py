from src.image_classification.simulation import load_train_data, load_model_to_train, train, save_model

loader, classes = load_train_data()
network, criterion, optimizer = load_model_to_train(len(classes))
network = train(network, loader, optimizer, criterion)
save_model(network, classes)
