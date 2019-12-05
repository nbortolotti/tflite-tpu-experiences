import pandas as pd
import numpy as np

import tensorflow as tf


train_ds_url = "http://download.tensorflow.org/data/iris_training.csv"
test_ds_url = "http://download.tensorflow.org/data/iris_test.csv"
ds_columns = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Plants']
species = np.array(['Setosa', 'Versicolor', 'Virginica'], dtype=np.object)

#Load data
categories = 'Plants'

train_path = tf.keras.utils.get_file(train_ds_url.split('/')[-1], train_ds_url)
test_path = tf.keras.utils.get_file(test_ds_url.split('/')[-1], test_ds_url)

train = pd.read_csv(train_path, names=ds_columns, header=0)
train_plantfeatures, train_categories = train, train.pop(categories)

test = pd.read_csv(test_path, names=ds_columns, header=0)
test_plantfeatures, test_categories = test, test.pop(categories)

y_categorical = tf.keras.utils.to_categorical(train_categories, num_classes=3)
y_categorical_test = tf.keras.utils.to_categorical(test_categories, num_classes=3)


#build dataset
#def build_dataset():

dataset = tf.data.Dataset.from_tensor_slices((train_plantfeatures.values, y_categorical))
dataset = dataset.batch(32)
dataset = dataset.shuffle(1000)
dataset = dataset.repeat()

dataset_test = tf.data.Dataset.from_tensor_slices((test_plantfeatures.values, y_categorical_test))
dataset_test = dataset_test.batch(32)
dataset_test = dataset_test.shuffle(1000)
dataset_test = dataset_test.repeat()


#build model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(16, input_dim=4),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax),
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#train model

model.fit(dataset, steps_per_epoch=32, epochs=100, verbose=1)


#eval model
loss, accuracy = model.evaluate(dataset_test, steps=32)

print("loss:%f"% (loss))
print("accuracy: %f"%   (accuracy))

# predict
new_specie = np.array([7.9,3.8,6.4,2.0])
predition = np.around(model.predict(np.expand_dims(new_specie, axis=0))).astype(np.int)[0]
print(model.predict(np.expand_dims(new_specie, axis=0)))
print("This species should be %s" % species[predition.astype(np.bool)][0])

model.predict(np.expand_dims(new_specie, axis=0))

# traditional saving model [option 1]
#model.save('iris_model.h5')

#converting to tflite from keras [option 2]
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
#open ("iris_lite.tflite" , "wb") .write(tflite_model)


