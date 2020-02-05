import numpy as np
import PIL.Image as Image
import matplotlib.pylab as plt
import time

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

def main():
    classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"

    IMAGE_SHAPE = (224, 224)

    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
    ])


    img_file = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/demostration_images/2.jpg')
    img = Image.open(img_file).resize(IMAGE_SHAPE)

    img_array = np.array(img)/255.0
    #img_array.shape

    result = classifier.predict(img_array[np.newaxis, ...])
    result.shape

    predicted_class = np.argmax(result[0], axis=-1)
    predicted_class

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    # plt.imshow(img_array)
    # plt.axis('off')
    # predicted_class_name = imagenet_labels[predicted_class]
    # _ = plt.title("Prediction: " + predicted_class_name.title())
    # plt.show()

    inferenceTime(img_array,classifier ,imagenet_labels)


# explore time to do the inference
def inferenceTime(image, mClassifier, mLabels):
    start = time.time()
    result = mClassifier.predict(image[np.newaxis, ...])
    end = time.time()
    print(end - start)

    #predicted_class = np.argmax(result[0], axis=-1)
    #predicted_class_name = mLabels[predicted_class]




if __name__ == '__main__':
  main()
