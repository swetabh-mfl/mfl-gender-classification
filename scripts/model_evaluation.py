from tensorflow.keras.models import load_model
from scripts.data_preparation import prepare_data

def evaluate_model(validation_dir, model_path):
    _, validation_generator = prepare_data('', validation_dir)
    model = load_model(model_path)

    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation accuracy: {accuracy*100:.2f}%')

if __name__ == "__main__":
    validation_dir = 'path_to_validation_directory'
    model_path = 'model/gender_classification_model.h5'
    evaluate_model(validation_dir, model_path)

