#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import argparse
import tensorrt as trt
import time

from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import makeVideo
import torch.backends.cudnn as cudnn
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4
from libs import Transfer
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r31.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r31.pth',
                    help='pre-trained decoder path')
parser.add_argument("--style", default="data/style/in2.jpg",
                    help='path to style image')
parser.add_argument("--matrixPath", default="models/r31.pth",
                    help='path to pre-trained model')
parser.add_argument("--layer",default="r31",
                    help="features of which layer to transfer")

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print(opt)

################# MODEL #################
if(opt.layer == 'r31'):
    matrix = MulLayer(layer='r31')
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    matrix = MulLayer(layer='r41')
    vgg = encoder4()
    dec = decoder4()
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))
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
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()

    style = style.cuda()
    content = content.cuda()

cap = cv2.VideoCapture('data/videos/tram.avi')   #assume it's 576x1024 (HxW)
#cap.set(3,600)
#cap.set(4,800)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('data/videos/out_vid.avi', fourcc, 20.0, (1024,576))

style_tmp = cv2.imread('data/style/27.jpg')
style_tmp = style_tmp.transpose((2,0,1))
style_tmp = torch.from_numpy(style_tmp).unsqueeze(0)
style.data.copy_(style_tmp)
style = style / 255.0
with torch.no_grad():
    sF = vgg(style)

i = 0
tt = time.time()
while(True):
    ret, frame = cap.read()
    if not ret: break
    print(frame.shape)
    frame = frame.transpose((2,0,1))
    print(frame.shape)
    frame = torch.from_numpy(frame).unsqueeze(0)
    content.data.copy_(frame)
    content = content/255.0

    with torch.no_grad():
        cF = vgg(content)
        print(cF.shape, sF.shape)
        if(opt.layer == 'r41'):
            feature = matrix(cF[opt.layer],sF[opt.layer])
        else:
            feature = matrix(cF,sF)
        transfer = dec(feature)
        transfer = transfer.clamp(0,1).squeeze(0)*255
        transfer = transfer.type(torch.uint8).data.cpu().numpy()
        transfer = transfer.transpose((1,2,0))

        out.write(transfer)
        cv2.imshow('frame',transfer)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    i += 1
    print((time.time()-tt)/i)

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()
