#!/usr/bin/env python3
import torch
from libs.Matrix import MulLayer
from libs.models import encoder3, encoder4, encoder5
from libs.models import decoder3, decoder4, decoder5
from libs.Transfer import Transfer


encoders = [encoder3, encoder4, encoder5]
decoders_small = [decoder3]
decoders_big = [decoder4, decoder5]



for m in encoders:
    print('start', m)
    x = torch.randn((1, 3, 1920, 1080)).cuda()
    model = m().eval().cuda()
    torch.onnx.export(model, x, f"models/onnx/{str(m)}.onnx")
    del model
    print(m, 'finished')
for m in decoders_small:
    print('start', m)
    x = torch.ones((1, 256, 100, 100)).cuda()
    model = m().eval().cuda()
    torch.onnx.export(model, x, f"models/onnx/{str(m)}.onnx")
    del model
    print(m, 'finished')
for m in decoders_big:
    print('start', m)
    x = torch.ones((1, 512, 100, 100)).cuda()
    model = m().eval().cuda()
    torch.onnx.export(model, x, f"models/onnx/{str(m)}.onnx")
    del model
    print(m, 'finished')


for l in ['r31', 'r41']:
    if l == 'r31':
        C = 256
    else:
        C = 512
    x = torch.ones((1, C, 480, 270)).cuda()
    y = torch.ones((1, C, 480, 270)).cuda()
    model = MulLayer(layer=l).eval().cuda()
    torch.onnx.export(model, (x, y), f"models/onnx/mul_{l}.onnx")
    del model
    print(f'MulLayer {l} finished')

m = Transfer().eval().cuda()
x = torch.randn((1, 3, 1024, 576)).cuda()
y = torch.randn((1, 3, 1024, 576)).cuda()
torch.onnx.export(m, (x, y), f'models/onnx/transfer_r31.onnx')
print('Full transfer_r31 finished')


