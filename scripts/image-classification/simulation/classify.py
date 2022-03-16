from src.image_classification.simulation import load_test_data, load_trained_model, classify

loader = load_test_data()
model = load_trained_model()
results = classify(loader, model, showtxt=True)
