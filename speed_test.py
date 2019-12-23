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
from libs.Criterion import LossCriterion
from libs.utils import makeVideo
from libs import shufflenetv2

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from libs.models import encoder5

RGB = '-rgb' in sys.argv

WIDTH = 0.5
LOSS_MODULE_PATH = 'models/regular/vgg_r51.pth'

STYLE_PATH  = 'data/style/1024x576/27.jpg'

################# MODEL #################
vgg5 = encoder5()
vgg5.load_state_dict(torch.load(LOSS_MODULE_PATH))
vgg5.cuda().eval()


enc = shufflenetv2.shufflenet_v2_x1_encoder()
dec = shufflenetv2.shufflenet_v2_x1_decoder()
enc.load_state_dict(torch.load('models/regular/shufflenetv2_x1_encoder.pth'))
dec.load_state_dict(torch.load('models/regular/shufflenetv2_x1_decoder.pth'))
enc.eval().cuda()
dec.eval().cuda()
#summary(mm,(3,1024,576))

mat = shufflenetv2.MulLayer()
mat.eval().cuda()

################# GLOBAL VARIABLE #################
content = torch.Tensor(1,3,576,1024).cuda()

################# GPU  #################
cap = cv2.VideoCapture('data/videos/tram.avi')   #assume it's 576x1024 (HxW)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('data/videos/out_vid.avi', fourcc, 20.0, (1024,576))


criterion = LossCriterion(style_layers = ['r11','r21','r31', 'r41'],
                          content_layers=['r41'],
                          style_weight=0.02,
                          content_weight=1.0)


################## STYLE ####################3
style = cv2.imread(STYLE_PATH)
if RGB:
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
style = style.transpose((2,0,1))
style = torch.from_numpy(style).unsqueeze(0).cuda()
style = style / 255.0
with torch.no_grad():
    sF = enc(style)
    sF_loss = vgg5(style)

i = 0
tt = time()
with torch.no_grad():
    while(True):
        ret, frame = cap.read()
        if not ret: break
        if RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose((2,0,1))
        frame = torch.from_numpy(frame).unsqueeze(0)
        content.data.copy_(frame)
        content = content/255.0

        #torch.cuda.synchronize()
        xx = time()
        transfer = enc(content)
        #transfer = mat(transfer, sF)
        transfer = dec(transfer)
        #torch.cuda.synchronize()
        print('test model', time() - xx, transfer.shape)
        #cF = vgg_c(content)
        #torch.cuda.synchronize()
        #feature = matrix(cF,sF,0.5)
        #transfer = dec(feature)
#        transfer = dec(cF)

        if '-l' in sys.argv:
            cF_loss = vgg5(content)
            tF = vgg5(transfer)
            loss,styleLoss,contentLoss = criterion(tF,sF_loss,cF_loss)
            print(loss.item(), styleLoss.item(), contentLoss.item())
        transfer = transfer.clamp(0,1).squeeze(0)*255
        transfer = transfer.type(torch.uint8).data.cpu().numpy()
        transfer = transfer.transpose((1,2,0))
        if RGB:
            transfer = cv2.cvtColor(transfer, cv2.COLOR_BGR2RGB)


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
