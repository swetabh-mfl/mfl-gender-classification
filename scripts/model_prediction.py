import argparse
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image

def predict_gender(model_path, img_path):
    # Load the model with the custom KerasLayer
    model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target_size to match your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required

    # Predict gender
    prediction = model.predict(img_array)
    gender = 'Female' if prediction[0][0] > 0.5 else 'Male'  # Adjust based on your model's output format
    return gender

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict gender from an image')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('img_path', type=str, help='Path to the image file')

    args = parser.parse_args()
    gender = predict_gender(args.model_path, args.img_path)
    print(f'Predicted Gender: {gender}')