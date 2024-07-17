import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

def predict_gender(model_path, img_path):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Female"
    else:
        return "Male"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict gender from an image')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('img_path', type=str, help='Path to the image to be predicted')

    args = parser.parse_args()
    gender = predict_gender(args.model_path, args.img_path)
    print(f'Predicted Gender: {gender}')