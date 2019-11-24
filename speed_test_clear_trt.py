#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import time
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import onnx_tensorrt.backend as backend

LAYER = 'r31'

class ModelData(object):
    MODEL_PATH = 'models/onnx/transfer_r31.onnx'
    INPUT_SHAPE = (2, 3, 576, 1024)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    return h_input, d_nput, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

def build_engine_onnx(model_file):
    #with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    #    builder.max_workspace_size = 1 << 40
    #    #builder.max_workspace_size = common.GiB(8)
    #    # Load the Onnx model and parse it in order to populate the TensorRT network.
    #    with open(model_file, 'rb') as model:
    #        flag = parser.parse(model.read())
    #        print(flag)
    #    network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
    #    #network.add_input('input_0', ModelData.DTYPE, (2,3,1024,576))
    #    #print(network.get_input(0))
    #    eng = builder.build_cuda_engine(network)
    #    eng.max_batch_size = 2
    #    print(type(a))
    #    return a 
    model = onnx.laod(model_file)
    engine = backend.prepare(model, device='CUDA:1')

def load_normalized_test_case(image, pagelocked_buffer):
    b, c, h, w = ModelData.INPUT_SHAPE
    image = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
    image = (image / 255.0 - 0.45) / 0.225
    np.copyto(pagelocked_buffer, image)
    return image

def main():
    content_cap = cv2.VideoCapture('data/videos/tram.avi')   #assume it's 800x600 (HxW)
    style_tmp = cv2.imread('data/style/27.jpg')
    with build_engine_onnx(ModelData.MODEL_PATH) as engine:
        #h_content, d_content, h_style, d_style, h_output, d_output, stream = allocate_buffers(engine)
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            style_img = load_normalized_test_case(style_tmp, h_style)
            i = 0
            tt = time.time()
            while True:
                ret, frame = cap.read()
                if not ret: break
                content_img = load_normalized_test_case(frame, h_content)
                do_inference(context, h_input, d_input, h_output, d_output, stream)
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

