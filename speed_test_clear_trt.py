#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import time
import tensorrt as trt

LAYER = 'r31'

################# PREPARATIONS #################
with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open('models/onnx/transfer_r31.onnx', 'rb') as model:
        parser.parse(model.read())
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.

with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.build_cuda_engine(network, config) as engine:
    h_content = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.float32)
    h_style = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.float32)
    h_output = cuda.pagelocked_empty(engine.get_binding_shape(2).volume(), dtype=np.float32)

    # Allocate device memory for inputs and outputs.
    d_content = cuda.mem_alloc(h_content.nbytes)
    d_style = cuda.mem_alloc(h_style.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    with engine.create_execution_context() as context:
		# Transfer input data to the GPU.
		cuda.memcpy_htod_async(d_content, h_content, stream)
		cuda.memcpy_htod_async(d_style, h_style, stream)
		# Run inference.
		context.execute_async(bindings=[int(d_content), int(d_style), int(d_output)], stream_handle=stream.handle)
		# Transfer predictions back from the GPU.
		cuda.memcpy_dtoh_async(h_output, d_output, stream)
		# Synchronize the stream
		stream.synchronize()
		# Return the host output.
    return h_output

################# GLOBAL VARIABLE #################
content = torch.Tensor(1,3,1920,1080)
style = torch.Tensor(1,3,1920,1080)

################# I/O ##################
cap = cv2.VideoCapture('/home/kamil/Desktop/style-transfer-net/data/videos/in/in_vid.mp4')   #assume it's 800x600 (HxW)
#cap.set(3,600)
#cap.set(4,800)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('/home/kamil/Desktop/style-transfer-net/data/videos/out/out_vid.avi', fourcc, 20.0, (600,800))

style_tmp = cv2.imread('/home/kamil/Desktop/style-transfer-net/data/style/style1.jpg')
style_tmp = style_tmp.transpose((2,0,1))
style_tmp = torch.from_numpy(style_tmp).unsqueeze(0)
style.data.copy_(style_tmp)
style = style / 255.0
with torch.no_grad():
    sF = vgg(style)


############# main #####################
i = 0
tt = time.time()
while(True):
    ret, frame = cap.read()
    if not ret: break
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame).unsqueeze(0)
    content.data.copy_(frame)
    content = content/255.0

    with torch.no_grad():
        cF = vgg(content)
        if(LAYER == 'r41'):
            feature = matrix(cF[LAYER], sF[LAYER])
        else:
            feature = matrix(cF, sF)
        transfer = dec(feature)
    transfer = transfer.clamp(0,1).squeeze(0)*255
    transfer = transfer.type(torch.uint8).data.cpu().numpy()
    transfer = transfer.transpose((1,2,0))

    #out.write(np.uint8(transfer*255))
    #cv2.imshow('frame',transfer)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    i += 1
    print((time.time()-tt)/i)

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()

