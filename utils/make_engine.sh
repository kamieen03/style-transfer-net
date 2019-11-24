#!/usr/bin/env bash
./utils/compile_full_model.py

python3 -m onnxsim models/onnx/vgg_r31.onnx models/onnx/_vgg_r31.onnx
rm models/onnx/vgg_r31.onnx
mv models/onnx/_vgg_r31.onnx models/onnx/vgg_r31.onnx

python3 -m onnxsim models/onnx/matrix_r31.onnx models/onnx/_matrix_r31.onnx
rm models/onnx/matrix_r31.onnx
mv models/onnx/_matrix_r31.onnx models/onnx/matrix_r31.onnx

python3 -m onnxsim models/onnx/decoder_r31.onnx models/onnx/_decoder_r31.onnx
rm models/onnx/decoder_r31.onnx
mv models/onnx/_decoder_r31.onnx models/onnx/decoder_r31.onnx

onnx2trt models/onnx/vgg_r31.onnx -o models/trt/vgg_r31.trt -v -w 2147484000 -b 1
onnx2trt models/onnx/matrix_r31.onnx -o models/trt/matrix_r31.trt -v -w 2147484000 -b 1
onnx2trt models/onnx/decoder_r31.onnx -o models/trt/decoder_r31.trt -v -w 2147484000 -b 1

