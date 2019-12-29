#!/usr/bin/env python3
import sys
import os
import torch
import onnx
sys.path.append(os.path.abspath(__file__ + "/../../"))

from libs.parametric_models import encoder3, decoder3, MulLayer

V2 = False
WIDTH = 0.25

e3c = encoder3(WIDTH, V2).eval().cuda()
e3s = encoder3(WIDTH, V2).eval().cuda()
matrix = MulLayer(WIDTH).eval().cuda()
d3 =  decoder3(WIDTH, V2).eval().cuda()

e3c.load_state_dict(torch.load('models/pruned/vgg_c_r31.pth'))
e3s.load_state_dict(torch.load('models/pruned/vgg_s_r31.pth'))
matrix.load_state_dict(torch.load('models/pruned/matrix_r31.pth'))
d3.load_state_dict(torch.load('models/pruned/dec_r31.pth'))

if torch.no_grad():
    x = torch.ones(1,3,1024,576).cuda()
    torch.onnx.export(e3c, x, 'models/onnx/vgg_c.onnx', verbose=True,
                        input_names=['input_vgg'], output_names=['output_vgg'])
    print("Saved encoder")

    del x
    x = torch.ones(1,3,1024,576).cuda()
    torch.onnx.export(e3s, x, 'models/onnx/vgg_s.onnx', verbose=True,
                        input_names=['input_vgg'], output_names=['output_vgg'])
    print("Saved encoder")
    del x

    x = torch.ones(1,64,256,144).cuda()
    y = torch.ones(1,64,256,144).cuda()
    torch.onnx.export(matrix, (x, y), 'models/onnx/matrix.onnx', verbose=True,
                        input_names=['content', 'style'], output_names=['output_matrix'])
    print("Saved matrix")
    del x; del y

    x = torch.ones(1,64,256,144).cuda()
    torch.onnx.export(d3, x, 'models/onnx/decoder.onnx', verbose=True,
                        input_names=['input_decoder'], output_names=['output_decoder'])
    print("Saved decoder")
