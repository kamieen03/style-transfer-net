#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import time
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit

VISUAL = False

class ModelData(object):
    MODEL_PATH = 'models/trt/transfer_r31.trt'
    INPUT_SHAPE = (1, 3, 576, 1024)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    h_content = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_style = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(2)), dtype=trt.nptype(ModelData.DTYPE))

    d_content = cuda.mem_alloc(h_content.nbytes)
    d_style = cuda.mem_alloc(h_style.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    return h_content, d_content, h_style, d_style, h_output, d_output, stream

def do_inference(context, h_content, d_content, h_style, d_style, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_content, h_content, stream)
    # pass both content and style through the encoder every time
    context.execute_async(bindings=[int(d_content), int(d_style), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

def build_engine_onnx(model_file):
    with open(model_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def load_normalized_test_case(image, pagelocked_buffer):
    image = image.transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
    image = image / 255.0
    np.copyto(pagelocked_buffer, image)
    return image

def main():
    content_cap = cv2.VideoCapture('data/videos/tram.avi')   #assume it's 800x600 (HxW)
    style_tmp = cv2.imread('data/style/1024x576/27.jpg')
    with build_engine_onnx(ModelData.MODEL_PATH) as engine:
        h_content, d_content, h_style, d_style, h_output, d_output, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            style_img = load_normalized_test_case(style_tmp, h_style)
            cuda.memcpy_htod_async(d_style, h_style, stream)
            i = 0
            tt = time.time()
            NEW_STYLE = True
            while True:
                ret, frame = content_cap.read()
                if not ret: break
                content_img = load_normalized_test_case(frame, h_content)
                do_inference(context, h_content, d_content, h_style, d_style, h_output, d_output, stream)
                i += 1
                out = np.reshape(h_output, (3, 576, 1024))
                out = out.clip(0,1)*255
                out = out.astype('uint8')
                out = out.transpose((1,2,0))
                print((time.time()-tt)/i)

                #out.write(np.uint8(transfer*255))
                if VISUAL:
                    cv2.imshow('frame', out)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


#while(True):
#    transfer = transfer.clamp(0,1).squeeze(0)*255
#    transfer = transfer.type(torch.uint8).data.cpu().numpy()
#    transfer = transfer.transpose((1,2,0))

