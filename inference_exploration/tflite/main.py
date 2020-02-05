import collections
import operator
import numpy as np
import time
from PIL import Image

import tflite_runtime.interpreter as tflite


EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

def load_labels(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
          return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
          pairs = [line.split(' ', maxsplit=1) for line in lines]
          return {int(index): label.strip() for index, label in pairs}
        else:
          return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def input_size():
  #using a 224 picture size
  return 224, 224


def input_tensor(interpreter):
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter):
  output_details = interpreter.get_output_details()[0]
  output_data = np.squeeze(interpreter.tensor(output_details['index'])())
  scale, zero_point = output_details['quantization']
  return scale * (output_data - zero_point)


def set_input(interpreter, data):
  input_tensor(interpreter)[:, :] = data


def get_output(interpreter, top_k=1, score_threshold=0.0):
  Class = collections.namedtuple('Class', ['id', 'score'])
  scores = output_tensor(interpreter)
  classes = [
      Class(i, scores[i])
      for i in np.argpartition(scores, -top_k)[-top_k:]
      if scores[i] >= score_threshold
  ]
  return sorted(classes, key=operator.itemgetter(1), reverse=True)


def main():
    labels = load_labels("models/imagenet_labels.txt")

    interpreter = make_interpreter("models/mobilenet_v2_1.0_224_quant_edgetpu.tflite")
    interpreter.allocate_tensors()

    size = input_size()
    image = Image.open("images/2.jpg").convert('RGB').resize(size, Image.ANTIALIAS)
    set_input(interpreter, image)

    # 5 times to compare inferences and also calculate after second inference
    for _ in range(5):
        start = time.monotonic()
        interpreter.invoke()
        inference_time = time.monotonic() - start
        classes = get_output(interpreter, 1, 0.0)
        print('%.1fms' % (inference_time * 1000))

    print('Results:')
    for klass in classes:
        print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))


if __name__ == '__main__':
  main()