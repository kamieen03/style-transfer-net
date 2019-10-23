#!/usr/bin/env python3
import torch
from torch2trt import torch2trt


from libs.models import encoder3, encoder4, encoder5
from libs.models import decoder3, decoder4, decoder5
from libs.Matrix import MulLayer

encoders = [encoder3, encoder4, encoder5]
decoders_small = [decoder3]
decoders_big = [decoder4, decoder5]

for m in encoders:
    print('start', m)
    x = torch.ones((1, 3, 1920, 1080)).cuda()
    model = m().eval().cuda()
    model_trt = torch2trt(model, [x])
    torch.save(model_trt.state_dict(), f'models/tensorrt/{str(m)}')
    del model
    del model_trt
    print(m, 'finished')
for m in decoders_small:
    print('start', m)
    x = torch.ones((1, 256, 100, 100)).cuda()
    model = m().eval().cuda()
    model_trt = torch2trt(model, [x])
    torch.save(model_trt.state_dict(), f'models/tensorrt/{str(m)}')
    del model
    del model_trt
    print(m, 'finished')
for m in decoders_big:
    print('start', m)
    x = torch.ones((1, 512, 100, 100)).cuda()
    model = m().eval().cuda()
    model_trt = torch2trt(model, [x])
    torch.save(model_trt.state_dict(), f'models/tensorrt/{str(m)}')
    del model
    del model_trt
    print(m, 'finished')

#TODO: MulLayer doesny work
for l in ['r31', 'r41']:
    x = torch.ones((1, 256, 480, 270)).cuda()
    y = torch.ones((1, 256, 480, 270)).cuda()
    model = MulLayer(layer=l).eval().cuda()
    model_trt = torch2trt(model, [x, y])
    torch.save(model_trt.state_dict(), f'models/tensorrt/mul_{l}.pth')
    del model
    del model_trt
    print(m, 'finished')

