import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_model(num_classes):
    # Load the pre-trained model from TensorFlow Hub
    module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
    hub_layer = hub.KerasLayer(module_url, trainable=False, name="tf_hub_layer")

    # Example additional layers
    inputs = Input(shape=(224, 224, 3))
    x = hub_layer(inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="gender_classification_model")
    return model