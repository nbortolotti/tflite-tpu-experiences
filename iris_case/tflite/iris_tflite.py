import numpy as np
import tflite_runtime.interpreter as tflite


interpreter = tflite.Interpreter(model_path="converted_model.tflite") # change the tflite model
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