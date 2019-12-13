#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import argparse
import tensorrt as trt
from time import time
from torchsummary import summary
import sys

from libs.Loader import Dataset
from libs.utils import makeVideo
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

PARAMETRIC = False
if len(sys.argv) > 1 and sys.argv[1] == '-p':
    PARAMETRIC = True

if PARAMETRIC:
    from libs.parametric_models import encoder3, decoder3, MulLayer
else:
    from libs.models import encoder3
    from libs.models import decoder3
    from libs.Matrix import MulLayer


if PARAMETRIC:
    VGG_PATH    = 'models/parametric/vgg_r31.pth'
    DEC_PATH    = 'models/parametric/dec_r31.pth'
    MATRIX_PATH = 'models/parametric/matrix_r31.pth'
else:
    VGG_PATH    = 'models/regular/vgg_r31.pth'
    DEC_PATH    = 'models/regular/dec_r31.pth'
    MATRIX_PATH = 'models/regular/r31.pth'

STYLE_PATH  = 'data/style/27.jpg'
LAYER = 'r31'
WIDTH = 0.5

################# MODEL #################
if(LAYER == 'r31'):
    matrix = MulLayer(layer='r31', WIDTH)
    vgg = encoder3(WIDTH)
    dec = decoder3(WIDTH)
elif(LAYER == 'r41'):
    matrix = MulLayer(layer='r41')
    vgg = encoder4()
    dec = decoder4()
vgg.load_state_dict(torch.load(VGG_PATH))
dec.load_state_dict(torch.load(DEC_PATH))
#matrix.load_state_dict(torch.load(MATRIX_PATH))
for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False

################# GLOBAL VARIABLE #################
content = torch.Tensor(1,3,576,1024)
style = torch.Tensor(1,3,256,256)

################# GPU  #################
vgg.cuda().eval()
dec.cuda().eval()
matrix.cuda().eval()

style = style.cuda()
content = content.cuda()

#summary(vgg, (3,576,1024))
#summary(matrix, [(256,144,256), (256, 64, 64)])
#summary(dec, (256,144,256))

cap = cv2.VideoCapture('data/videos/tram.avi')   #assume it's 576x1024 (HxW)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('data/videos/out_vid.avi', fourcc, 20.0, (1024,576))

style_tmp = cv2.imread(STYLE_PATH)
style_tmp = style_tmp.transpose((2,0,1))
style_tmp = torch.from_numpy(style_tmp).unsqueeze(0)
style.data.copy_(style_tmp)
style = style / 255.0
with torch.no_grad():
    sF = vgg(style)

i = 0
tt = time()
while(True):
    ret, frame = cap.read()
    if not ret: break
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame).unsqueeze(0)
    content.data.copy_(frame)
    content = content/255.0

    with torch.no_grad():
        cF = vgg(content)
        torch.cuda.synchronize()
        #feature = matrix(cF,sF)
        #transfer = dec(feature)
        transfer = dec(cF)
        transfer = transfer.clamp(0,1).squeeze(0)*255
        transfer = transfer.type(torch.uint8).data.cpu().numpy()
        transfer = transfer.transpose((1,2,0))

        #out.write(transfer)
        cv2.imshow('frame',transfer)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    i += 1
    print((time()-tt)/i)

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()
