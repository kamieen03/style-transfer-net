#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))

SAVE_PTH = True
SAVE_ONNX = True

from libs.Transfer import Transfer
from libs.models import encoder3 
import torch
import onnx

model = Transfer()
model.vgg.load_state_dict(torch.load('models/vgg_r31.pth'))
model.dec.load_state_dict(torch.load('models/dec_r31.pth'))
model.matrix.load_state_dict(torch.load('models/r31.pth'))
model.cuda().eval()

x = torch.ones(2,3,1024,576).cuda()
if SAVE_PTH:
    torch.save(model.state_dict(), 'models/transfer_r31.pth')
    print("Saved pth model")
if SAVE_ONNX:
    torch.onnx.export(model, x, 'models/onnx/transfer_r31.onnx', verbose=True,
                        input_names=['input_0'], output_names=['output_0'])
    print("Saved onnx model")
