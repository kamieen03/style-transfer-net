#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

import cv2
import torch
import numpy as np
import time

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from calibrator import TransferEntropyCalibrator


VISUAL = True

VGG_PATH = 'models/trt/vgg_r31.trt'
MATRIX_PATH = 'models/trt/matrix_r31.trt'
DECODER_PATH = 'models/trt/decoder_r31.trt'
STYLE_PATH = 'data/style/1024x576/27.jpg'
CONTENT_VID_PATH = 'data/videos/tram.avi'

DTYPE = trt.float32
TRT_LOGGER = trt.Logger()

vgg_engine = None
matrix_engine = None
decoder_engine = None

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def GiB(val):
    return val * 1 << 30

# Allocate host and device buffers, and create a stream.
def allocate_buffers(vgg_engine, matrix_engine, decoder_engine):
    h_content = cuda.pagelocked_empty(trt.volume(vgg_engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_style = cuda.pagelocked_empty(trt.volume(vgg_engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_vgg_content_out = cuda.pagelocked_empty(trt.volume(vgg_engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    h_vgg_style_out = cuda.pagelocked_empty(trt.volume(vgg_engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
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
    with vgg_engine.create_execution_context() as context:
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
        out = np.reshape(h_decoder_out, (3, 576, 1024))
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


def main():
    # deserialize trt engines
    global vgg_engine, matrix_engine, decoder_engine
    with trt.Runtime(TRT_LOGGER) as runtime:
        with open(VGG_PATH, 'rb') as f:
            vgg_engine = runtime.deserialize_cuda_engine(f.read())
        with open(MATRIX_PATH, 'rb') as f:
            matrix_engine = runtime.deserialize_cuda_engine(f.read())
        with open(DECODER_PATH, 'rb') as f:
            decoder_engine = runtime.deserialize_cuda_engine(f.read())

    # allocate buffers
    h_content, d_content, h_style, d_style, h_vgg_content_out, d_vgg_content_out, \
    h_vgg_style_out, d_vgg_style_out, h_matrix_out, d_matrix_out, \
    h_decoder_out, d_decoder_out, stream = allocate_buffers(vgg_engine, matrix_engine, decoder_engine)

    # encode style features, placing them in GPU memory
    style_tmp = cv2.imread(STYLE_PATH)
    style_img = load_normalized_test_case(style_tmp, h_style)
    cuda.memcpy_htod_async(d_style, h_style, stream)
    with vgg_engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_style), int(d_vgg_style_out)], stream_handle=stream.handle)

    # run inference on all video frames consecutively
    loop_inference(h_content, d_content, h_style, d_style,
            h_vgg_content_out, d_vgg_content_out, h_vgg_style_out, d_vgg_style_out,
            h_matrix_out, d_matrix_out, h_decoder_out, d_decoder_out, stream)

if __name__ == '__main__':
    main()



# This function builds an engine from a Caffe model.
def build_int8_encoder(deploy_file, model_file, calib, batch_size=32):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size
        builder.max_workspace_size = common.GiB(2)
        builder.int8_mode = True
        builder.int8_calibrator = calib
        # Parse Caffe model
        with open(VGG_PATH, 'rb') as model:
            parser.parse(model.read())
        # Build engine and do int8 calibration.
        return builder.build_cuda_engine(network)


def main():
    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    batch_size = 8
    calibration_cache = "calibration_cache/decoder_r31.txt"
    calib = TransferEntropyCalibrator(test_set, calibration_cache, 8)

    # Inference batch size can be different from calibration batch size.
    with build_int8_engine(deploy_file, model_file, calib, batch_size) as engine, engine.create_execution_context() as context:
        # Batch size for inference can be different than batch size used for calibration.
        check_accuracy(context, batch_size, test_set=load_mnist_data(test_set), test_labels=load_mnist_labels(test_labels))

if __name__ == '__main__':
    main()
