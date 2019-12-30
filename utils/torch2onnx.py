#!/usr/bin/env python3
import torch

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))

from libs.parametric_models import encoder3, decoder3, MulLayer

V2 = True
WIDTH = 0.25

e3c = encoder3(WIDTH, V2).eval().cuda()
e3s = encoder3(WIDTH, V2).eval().cuda()
mat = MulLayer(WIDTH).eval().cuda()
d3 =  decoder3(WIDTH, V2).eval().cuda()



e3c.load_state_dict(torch.load('models/pruned/vgg_c_r31.pth'))
e3s.load_state_dict(torch.load('models/pruned/vgg_s_r31.pth'))
matrix.load_state_dict(torch.load('models/pruned/matrix_r31.pth'))
dec.load_state_dict(torch.load('models/pruned/dec_r31.pth'))

with torch.no_grad():
    x = torch.randn((1, 3, 1024, 576)).cuda()
    torch.onnx.export(m, (x,), f'models/onnx/vgg_c_r31.onnx')
    torch.onnx.export(m, (x,), f'models/onnx/vgg_c_r31.onnx')
    print('Full transfer_r31 finished')


