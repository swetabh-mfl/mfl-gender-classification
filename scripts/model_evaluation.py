import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from data_preparation import prepare_test_data

def evaluate_model(model_path, test_dir, batch_size=32):
    # Load the model with custom_objects
    model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    test_generator = prepare_test_data(test_dir)

    results = model.evaluate(test_generator, batch_size=batch_size)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")

if __name__ == "__main__":
    model_path = 'models/gender_classification_model.h5'
    test_dir = 'data/test'
    batch_size = 32
    evaluate_model(model_path, test_dir, batch_size)