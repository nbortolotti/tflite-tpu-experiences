#!/bin/bash
readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly image_data_url=https://storage.googleapis.com/demostration_images/
readonly model_data_url=https://github.com/google-coral/edgetpu/raw/master/test_data/

#echo $script_dir
# Get example image
image_dir="${script_dir}/images"
#echo $image_dir
mkdir -p "${image_dir}"
#mkdir "${image_dir}"

(cd "${image_dir}" && curl -OL "${image_data_url}2.jpg")

# Get TF Lite model and labels
model_dir="${script_dir}/models"
mkdir -p "${model_dir}"

(cd "${model_dir}" && \
curl -OL "${model_data_url}mobilenet_v2_1.0_224_quant_edgetpu.tflite" \
     -OL "${model_data_url}imagenet_labels.txt")
