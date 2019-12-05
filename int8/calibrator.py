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

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np

import torch
from torchvision import transforms, datasets

                                            
# Returns a numpy buffer of shape (num_images, 1, 28, 28)
def load_datasets(batch_size):
    tr = transforms.Compose([
            transforms.Resize(1024,576),
            transforms.ToTensor()
        ])

    content_dataset = datasets.ImageFolder(root='../data/mscoco/train/', transform=tr)
    content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size)
    content_iter = iter(content_loader)
    style_dataset = datasets.ImageFolder(root='../data/wikiart/train/', transform=tr)
    style_loader = torch.utils.data.DataLoader(style_dataset, batch_size=batch_size)
    style_iter = iter(style_loader)
    return content_iter, style_iter
    #return np.ascontiguousarray((raw_buf[16:] / 255.0).astype(np.float32).reshape(num_images, image_c, image_h, image_w))


class TransferEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size=16):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.content_iter, self.style_iter = load_datasets(batch_size)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.content_buffer = cuda.mem_alloc(3*1024*576*4 * self.batch_size)
        self.style_buffer = cuda.mem_alloc(3*1024*576*4 * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        print(names)
        try:
            content = next(self.content_iter).numpy().ravel()
            style = next(self.style_iter).numpy().ravel()
        except StopIteration:
            return None, None
        cuda.memcpy_htod(self.content_buffer, content)
        cuda.memcpy_htod(self.style_buffer, style)
        return [self.content_buffer, self.style_buffer]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

class AbstractEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size=16):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.content_iter, self.style_iter = load_datasets(batch_size)
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class EncoderEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size=16):
        AbstractEntropyCalibrator.__init__(self, cache_file, batch_size)
        self.content_buffer = cuda.mem_alloc(3*1024*576*4 * self.batch_size)

    def get_batch(self, names):
        print(names)
        try:
            content = next(self.content_iter).numpy().ravel()
        except StopIteration:
            return None, None
        cuda.memcpy_htod(self.content_buffer, content)
        return [self.content_buffer]

