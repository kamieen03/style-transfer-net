#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import time
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit

LAYER = 'r31'

class ModelData(object):
    MODEL_PATH = '/home/kamil/Desktop/style-transfer-net/models/onnx/transfer_r31.onnx'
    INPUT_SHAPE = (3, 1920, 1080)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_content = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_style = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(2)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_content = cuda.mem_alloc(h_input.nbytes)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_content, d_content, h_style, d_style, h_output, d_output, stream

def do_inference(context, h_content, d_content, h_style, d_style, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_content, h_content, stream)
    cuda.memcpy_htod_async(d_style, h_style, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_content), int(d_style), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 20
        #builder.max_workspace_size = common.GiB(8)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        print(model_file)
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
        a = builder.build_cuda_engine(network)
        print(type(a))
        return a #TODO: a is None :(

def load_normalized_test_case(image, pagelocked_buffer):
    c, h, w = ModelData.INPUT_SHAPE
    image = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
    image = (image / 255.0 - 0.45) / 0.225
    np.copyto(pagelocked_buffer, image)
    return test_image

def main():
    content_cap = cv2.VideoCapture('/home/kamil/Desktop/style-transfer-net/data/videos/in/in_vid.mp4')   #assume it's 800x600 (HxW)
    style_tmp = cv2.imread('/home/kamil/Desktop/style-transfer-net/data/style/style1.jpg')
    with build_engine_onnx(ModelData.MODEL_PATH) as engine:
        h_content, d_content, h_style, d_style, h_output, d_output, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            style_img = load_normalized_test_case(style_tmp, h_style)

            i = 0
            tt = time.time()
            while True:
                ret, frame = cap.read()
                if not ret: break
                conetnt_img = load_normalized_test_case(frame, h_content)
                do_inference(context, h_content, d_content, h_style, d_style, h_output, d_output, stream)
                i += 1
                print((time.time()-tt)/i, type(h_output))
                #out.write(np.uint8(transfer*255))
                #cv2.imshow('frame',transfer)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


#while(True):
#    transfer = transfer.clamp(0,1).squeeze(0)*255
#    transfer = transfer.type(torch.uint8).data.cpu().numpy()
#    transfer = transfer.transpose((1,2,0))

