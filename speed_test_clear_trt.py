#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys

VISUAL = '-v' in sys.argv

VGG_C_PATH   = 'models/trt/vgg_c.trt'
VGG_S_PATH   = 'models/trt/vgg_s.trt'
MATRIX_PATH  = 'models/trt/matrix.trt'
DECODER_PATH = 'models/trt/decoder.trt'

STYLE_PATH       = 'data/style/mobile.jpg'
CONTENT_VID_PATH = 'data/videos/tram_mobile.avi'

DTYPE = trt.float32

vgg_c_engine = None
vgg_s_engine = None
matrix_engine = None
decoder_engine = None

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# Allocate host and device buffers, and create a stream.
def allocate_buffers(vgg_c_engine, vgg_s_engine, matrix_engine, decoder_engine):
    h_content = cuda.pagelocked_empty(trt.volume(vgg_c_engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_style = cuda.pagelocked_empty(trt.volume(vgg_s_engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_vgg_content_out = cuda.pagelocked_empty(trt.volume(vgg_c_engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    h_vgg_style_out = cuda.pagelocked_empty(trt.volume(vgg_s_engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    h_matrix_out = cuda.pagelocked_empty(trt.volume(matrix_engine.get_binding_shape(2)), dtype=trt.nptype(DTYPE))
    h_decoder_out = cuda.pagelocked_empty(trt.volume(decoder_engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))

    d_content = cuda.mem_alloc(h_content.nbytes)
    d_style = cuda.mem_alloc(h_style.nbytes)
    d_vgg_content_out = cuda.mem_alloc(h_vgg_content_out.nbytes)
    d_vgg_style_out = cuda.mem_alloc(h_vgg_style_out.nbytes)
    d_matrix_out = cuda.mem_alloc(h_matrix_out.nbytes)
    d_decoder_out = cuda.mem_alloc(h_decoder_out.nbytes)

    stream = cuda.Stream()
    return h_content, d_content, h_style, d_style, h_vgg_content_out, d_vgg_content_out, \
           h_vgg_style_out, d_vgg_style_out, h_matrix_out, d_matrix_out, \
           h_decoder_out, d_decoder_out, stream

def do_inference(h_content, d_content, h_style, d_style, h_vgg_content_out, d_vgg_content_out,
           h_vgg_style_out, d_vgg_style_out, h_matrix_out, d_matrix_out, 
           h_decoder_out, d_decoder_out, stream):

    # copy content img to GPU memory
    cuda.memcpy_htod_async(d_content, h_content, stream)

    # pass content through vgg
    with vgg_c_engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_content), int(d_vgg_content_out)], stream_handle=stream.handle)
    # pass content and style features through transformation module
    with matrix_engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_vgg_content_out), int(d_vgg_style_out),
            int(d_matrix_out)], stream_handle=stream.handle)
    # pass transformed features throgh decoder
    with decoder_engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_matrix_out), int(d_decoder_out)], stream_handle=stream.handle)
    # move stylized picture to host memory
    cuda.memcpy_dtoh_async(h_decoder_out, d_decoder_out, stream)
    stream.synchronize()


def load_normalized_test_case(image, pagelocked_buffer):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose([2, 0, 1]).astype(trt.nptype(DTYPE)).ravel()
    image = image / 255.0
    np.copyto(pagelocked_buffer, image)
    return image

def loop_inference(h_content, d_content, h_style, d_style,
            h_vgg_content_out, d_vgg_content_out, h_vgg_style_out, d_vgg_style_out,
            h_matrix_out, d_matrix_out, h_decoder_out, d_decoder_out, stream):
    i = 0
    tt = time.time()
    content_cap = cv2.VideoCapture(CONTENT_VID_PATH)

    while True:
        ret, frame = content_cap.read()
        if not ret: break
        content_img = load_normalized_test_case(frame, h_content)
        do_inference(h_content, d_content, h_style, d_style,
            h_vgg_content_out, d_vgg_content_out, h_vgg_style_out, d_vgg_style_out,
            h_matrix_out, d_matrix_out, h_decoder_out, d_decoder_out, stream)
        i += 1
        out = np.reshape(h_decoder_out, (3, 1024, 576))
        out = out.clip(0,1)*255
        out = out.astype('uint8')
        out = out.transpose((1,2,0))
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        print((time.time()-tt)/i)

        #out.write(np.uint8(transfer*255))
        if VISUAL:
            cv2.imshow('frame', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # deserialize trt engines
    global vgg_c_engine, vgg_s_engine, matrix_engine, decoder_engine
    with trt.Runtime(TRT_LOGGER) as runtime:
        with open(VGG_C_PATH, 'rb') as f:
            vgg_c_engine = runtime.deserialize_cuda_engine(f.read())
        with open(VGG_S_PATH, 'rb') as f:
            vgg_s_engine = runtime.deserialize_cuda_engine(f.read())
        with open(MATRIX_PATH, 'rb') as f:
            matrix_engine = runtime.deserialize_cuda_engine(f.read())
        with open(DECODER_PATH, 'rb') as f:
            decoder_engine = runtime.deserialize_cuda_engine(f.read())

    # allocate buffers
    h_content, d_content, h_style, d_style, h_vgg_content_out, d_vgg_content_out, \
    h_vgg_style_out, d_vgg_style_out, h_matrix_out, d_matrix_out, \
    h_decoder_out, d_decoder_out, stream = allocate_buffers(vgg_c_engine, vgg_s_engine,
                                                matrix_engine, decoder_engine)

    # encode style features, placing them in GPU memory
    style_tmp = cv2.imread(STYLE_PATH)
    style_img = load_normalized_test_case(style_tmp, h_style)
    cuda.memcpy_htod_async(d_style, h_style, stream)
    with vgg_s_engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_style), int(d_vgg_style_out)], stream_handle=stream.handle)

    # run inference on all video frames consecutively
    loop_inference(h_content, d_content, h_style, d_style,
            h_vgg_content_out, d_vgg_content_out, h_vgg_style_out, d_vgg_style_out,
            h_matrix_out, d_matrix_out, h_decoder_out, d_decoder_out, stream)

if __name__ == '__main__':
    main()

