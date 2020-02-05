import pytest
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from inference_exploration.cpu.main_cpu import image_analysis

def test_image():
    classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"
    image_shape = (224, 224)
    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=image_shape + (3,))
    ])

    img_file = tf.keras.utils.get_file('image.jpg', 'https://storage.googleapis.com/demostration_images/2.jpg')
    img = Image.open(img_file).resize(image_shape)

    img_array = np.array(img) / 255.0

    predicted_class = image_analysis(classifier, image_shape, img_array)
    assert predicted_class == 209