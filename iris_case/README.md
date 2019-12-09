# Iris Academic Model

The initial idea of this example proposes to represent the iris dataset in a ktf.keras model, then transform it to tflite format and then consume it from the device.

## Represent Iris into a TF & Keras model

### Build the model
```
model = tf.keras.Sequential([
  tf.keras.layers.Dense(16, input_dim=4),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax),
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

### Train & Eval
``` 
model.fit(dataset, steps_per_epoch=32, epochs=100, verbose=1)

loss, accuracy = model.evaluate(dataset_test, steps=32)
```

## convert to tflite format
*for the initial example I will not optimize the conversion.
``` 
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open ("iris_lite.tflite" , "wb") .write(tflite_model)
```

## Deploy and consume into Coral Device
*to config Coral device please check the information provided here.
``` 
interpreter = tflite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model
new_specie = np.array([7.9,3.8,6.4,2.0]) # general example to predict
input_data = np.array(np.expand_dims(new_specie, axis=0), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```