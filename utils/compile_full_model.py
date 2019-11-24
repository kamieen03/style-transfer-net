#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))

SAVE_PTH = False
SAVE_ONNX = True

from libs.Transfer import Transfer
from libs.models import encoder3, decoder3
from libs.Matrix import MulLayer
import torch
import onnx

vgg = encoder3()
matrix = MulLayer(layer='r31')
decoder = decoder3()
vgg.load_state_dict(torch.load('models/vgg_r31.pth'))
matrix.load_state_dict(torch.load('models/r31.pth'))
decoder.load_state_dict(torch.load('models/dec_r31.pth'))
vgg.cuda().eval()
matrix.cuda().eval()
decoder.cuda().eval()

#model = Transfer()
#model.vgg.load_state_dict(torch.load('models/vgg_r31.pth'))
#model.dec.load_state_dict(torch.load('models/dec_r31.pth'))
#model.matrix.load_state_dict(torch.load('models/r31.pth'))
#model.cuda().eval()

if SAVE_PTH:
    torch.save(model.state_dict(), 'models/transfer_r31.pth')
    print("Saved pth model")
if SAVE_ONNX:
    x = torch.ones(1,3,576,1024).cuda()
    torch.onnx.export(vgg, x, 'models/onnx/vgg_r31.onnx', verbose=True,
                        input_names=['input_vgg'], output_names=['output_vgg'])
    print("Saved vgg")
    del x
    x = torch.ones(1,256,144,256).cuda()
    y = torch.ones(1,256,144,256).cuda()
    torch.onnx.export(matrix, (x, y), 'models/onnx/matrix_r31.onnx', verbose=True,
                        input_names=['content', 'style'], output_names=['output_matrix'])
    print("Saved matrix")
    del x; del y
    x = torch.ones(1,256,144,256).cuda()
    torch.onnx.export(decoder, x, 'models/onnx/decoder_r31.onnx', verbose=True,
                        input_names=['input_decoder'], output_names=['output_decoder'])
    print("Saved decoder")
