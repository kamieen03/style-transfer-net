#!/usr/bin/env bash
./utils/compile_full_model.py

models=(vgg_c vgg_s matrix dec)

for model in ${models[*]}
do
    python3 -m onnxsim models/onnx/$model.onnx models/onnx/_$model.onnx
    rm models/onnx/$model.onnx
    mv models/onnx/_$model.onnx models/onnx/$model.onnx
    onnx2trt models/onnx/$model.onnx -o models/trt/$model.trt -v -w 2147484000 -b 1
done

