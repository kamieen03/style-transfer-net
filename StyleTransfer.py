#!/usr/bin/env python3

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
base_path = os.path.dirname(os.path.abspath(__file__))
trt_path = os.path.join(base_path, 'models/trt')
VGG_C_PATH   = os.path.join(trt_path, 'vgg_c.trt')
VGG_S_PATH   = os.path.join(trt_path, 'vgg_s.trt')
MATRIX_PATH  = os.path.join(trt_path, 'matrix.trt')
DECODER_PATH = os.path.join(trt_path, 'decoder.trt')

DTYPE = trt.float32
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class StyleTransfer:
    def __init__(self):
        # deserialize trt engines
        with trt.Runtime(TRT_LOGGER) as runtime:
            with open(VGG_C_PATH, 'rb') as f:
                self.vgg_c_engine = runtime.deserialize_cuda_engine(f.read())
            with open(VGG_S_PATH, 'rb') as f:
                self.vgg_s_engine = runtime.deserialize_cuda_engine(f.read())
            with open(MATRIX_PATH, 'rb') as f:
                self.matrix_engine = runtime.deserialize_cuda_engine(f.read())
            with open(DECODER_PATH, 'rb') as f:
                self.decoder_engine = runtime.deserialize_cuda_engine(f.read())

        # allocate buffers
        self._allocate_buffers(self.vgg_c_engine, self.vgg_s_engine, self.matrix_engine, self.decoder_engine)

    # Allocate host and device buffers, and create a stream.
    def _allocate_buffers(self, vgg_c_engine, vgg_s_engine, matrix_engine, decoder_engine):
        self.h_content = cuda.pagelocked_empty(trt.volume(vgg_c_engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
        self.h_style = cuda.pagelocked_empty(trt.volume(vgg_s_engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
        self.h_vgg_content_out = cuda.pagelocked_empty(trt.volume(vgg_c_engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
        self.h_vgg_style_out = cuda.pagelocked_empty(trt.volume(vgg_s_engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
        self.h_alpha = cuda.pagelocked_empty(trt.volume(matrix_engine.get_binding_shape(2)), dtype=trt.nptype(DTYPE))
        self.h_matrix_out = cuda.pagelocked_empty(trt.volume(matrix_engine.get_binding_shape(3)), dtype=trt.nptype(DTYPE))
        self.h_decoder_out = cuda.pagelocked_empty(trt.volume(decoder_engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))

        self.d_content = cuda.mem_alloc(self.h_content.nbytes)
        self.d_style = cuda.mem_alloc(self.h_style.nbytes)
        self.d_vgg_content_out = cuda.mem_alloc(self.h_vgg_content_out.nbytes)
        self.d_vgg_style_out = cuda.mem_alloc(self.h_vgg_style_out.nbytes)
        self.d_alpha = cuda.mem_alloc(self.h_alpha.nbytes)
        self.d_matrix_out = cuda.mem_alloc(self.h_matrix_out.nbytes)
        self.d_decoder_out = cuda.mem_alloc(self.h_decoder_out.nbytes)

        self.stream = cuda.Stream()

    def _load_normalized_test_case(self, image, pagelocked_buffer):
        image = image.astype(trt.nptype(DTYPE)).ravel()
        np.copyto(pagelocked_buffer, image)

    def set_style(self, style, alpha=1.0):
        """
        style - style image; numpy array 3x1024x576 (CxHxW), RGB, uint8
        alpha - stylization strengh in [0; 1] interval; fp32
        """
        # encode style
        self._load_normalized_test_case(style, self.h_style)
        cuda.memcpy_htod_async(self.d_style, self.h_style, self.stream)
        with self.vgg_s_engine.create_execution_context() as context:
            context.execute_async(bindings=[int(self.d_style), int(self.d_vgg_style_out)], stream_handle=self.stream.handle)

        # copy alpha to GPU memory
        np.copyto(self.h_alpha, alpha)
        cuda.memcpy_htod_async(self.d_alpha, self.h_alpha, self.stream)

    def stylize_frame(self, frame):
        """
        frame - image to stylize; numpy array 3x1024x576 (CxHxW), RGB, uint8
        """

        self._load_normalized_test_case(frame, self.h_content)

        # copy content img to GPU memory
        cuda.memcpy_htod_async(self.d_content, self.h_content, self.stream)

        # pass content through vgg
        with self.vgg_c_engine.create_execution_context() as context:
            context.execute_async(bindings=[int(self.d_content), int(self.d_vgg_content_out)], stream_handle=self.stream.handle)
        # pass content and style features through transformation module
        with self.matrix_engine.create_execution_context() as context:
            context.execute_async(bindings=[int(self.d_vgg_content_out), int(self.d_vgg_style_out), int(self.d_alpha),
                int(self.d_matrix_out)], stream_handle=self.stream.handle)
        # pass transformed features throgh decoder
        with self.decoder_engine.create_execution_context() as context:
            context.execute_async(bindings=[int(self.d_matrix_out), int(self.d_decoder_out)], stream_handle=self.stream.handle)
        # move stylized picture to host memory
        cuda.memcpy_dtoh_async(self.h_decoder_out, self.d_decoder_out, self.stream)
        self.stream.synchronize()

        out = self.h_decoder_out.astype('uint8')
        return np.reshape(out, (3, 1024, 576))

