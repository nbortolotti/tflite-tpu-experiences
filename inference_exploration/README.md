![Python app inference exploration](https://github.com/nbortolotti/tflite-tpu-experiences/workflows/Python%20app%20inference%20exploration/badge.svg?branch=master)
# Inference exploration
## tflite
*example executed using a Coral edge tpu accelerator*

to remember: tflite_runtime is a dependency for the tflite implementation.

example:

`pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0-cp37-cp37m-linux_armv7l.whl`

more details, [here](https://www.tensorflow.org/lite/guide/python)

Complete requeriments before run the example. Is included a file with a simple bach script to download 
the image to analize and download the model + label of imagenet 2.
 
`install_requirements.sh`

# CPU
Into the folder cpu is included a simple demostration to analyze a image using imagenet 2.
Is also included a routine to check the performance of the inference

`def inferenceTime()`

## Colab 
Into this folder is included a simple representation of the cpu python script to execute directly in Colab
