from src.image_classification.simulation import load_test_data, load_trained_model, infer

loader = load_test_data()
model = load_trained_model()
results = infer(loader, model, showtxt=True)
