#!/usr/bin/env python3

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

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from calibrator import EncoderEntropyCalibrator
from calibrator import MatrixEntropyCalibrator
from calibrator import DecoderEntropyCalibrator


VGG_PATH = 'models/onnx/vgg_r31.onnx'
MATRIX_PATH = 'models/trt/matrix_r31.onnx'
DECODER_PATH = 'models/trt/decoder_r31.onnx'

DTYPE = trt.float32
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def GiB(val):
    return val * 1 << 30

def build_int8_engine(model_file, cache_file, calib, batch_size=4):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = batch_size
        builder.max_workspace_size = GiB(2)
        builder.int8_mode = True
        builder.int8_calibrator = calib
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
        return builder.build_cuda_engine(network)


def main():
    batch_size = 4
    encoder_calib_cache = "calibration_cache/encoder_r31.txt"
    matrix_calib_cache = "calibration_cache/matrix_r31.txt"
    dec_calib_cache = "calibration_cache/decoder_r31.txt"
    encoder_engine_path = 'models/trt/encoder_r31_int8.engine'
    matrix_engine_path = 'models/trt/matrix_r31_int8.engine'
    decoder_engine_path = 'models/trt/decoder_r31_int8.engine'

    H, W = 576, 1024
    calibrator = EncoderEntropyCalibrator(encoder_calib_cache, H, W, batch_size)
    with build_int8_engine(VGG_PATH, encoder_calib_cache, calibrator, batch_size) as engine:
        with open(encoder_engine_path, 'wb+') as f:
            f.write(engine.serialize())
    calibrator = MatrixEntropyCalibrator(matrix_calib_cache, H, W, encoder_engine_path)
    with build_int8_engine(MATRIX_PATH, matrix_calib_cache, calibrator, batch_size) as engine:
        with open(matrix_engine_path, 'wb+') as f:
            f.write(engine.serialize())
    calibrator = DecoderEntropyCalibrator(DECODER_PATH, H, W, encoder_engine_path,
                    matrix_engine_path)
    with build_int8_engine(model_file, decoder_calib_cache, calibrator, batch_size) as engine:
        with open(decoder_engine_path, 'wb+') as f:
            f.write(engine.serialize())

if __name__ == '__main__':
    main()
