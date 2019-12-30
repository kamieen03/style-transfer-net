#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import argparse
#import tensorrt as trt
from time import time
import sys

from libs.Loader import Dataset
from libs.Criterion import LossCriterion
from libs.utils import makeVideo
from libs import shufflenetv2

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from libs.models import encoder5

RGB = '-rgb' in sys.argv
PARAMETRIC = '-p' in sys.argv

WIDTH = 0.25
LOSS_MODULE_PATH = 'models/regular/vgg_r51.pth'

STYLE_PATH  = 'data/style/1024x576/sketch.jpg'

################# MODEL #################
if PARAMETRIC:
    from libs.parametric_models import encoder3, decoder3, MulLayer
    e3c = encoder3(0.25).eval().cuda()
    e3s = encoder3(0.25).eval().cuda()
    d3 = decoder3(0.25).eval().cuda()
    mat3 = MulLayer(0.25).eval().cuda()
    e3c.load_state_dict(torch.load('models/pruned/vgg_c_r31.pth'))
    e3s.load_state_dict(torch.load('models/pruned/vgg_s_r31.pth'))
    d3.load_state_dict(torch.load('models/pruned/dec_r31.pth'))
    mat3.load_state_dict(torch.load('models/pruned/matrix_r31.pth'))
    #e3c.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/vgg_c_r31.pth'))
    #e3s.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/vgg_s_r31.pth'))
    #d3.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/dec_r31.pth'))
    #mat3.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/matrix_r31.pth'))

else:
    from libs.models import encoder3, decoder3 
    from libs.Matrix import MulLayer
    e3c = encoder3().eval().cuda()
    e3s = encoder3().eval().cuda()
    d3 = decoder3().eval().cuda()
    e3c.load_state_dict(torch.load('models/regular/vgg_r31.pth'))
    e3s.load_state_dict(torch.load('models/regular/vgg_r31.pth'))
    d3.load_state_dict(torch.load('models/regular/dec_r31.pth'))

    mat3 = MulLayer('r31').eval().cuda()
    mat3.load_state_dict(torch.load('models/regular/r31.pth'))

vgg5 = encoder5()
vgg5.load_state_dict(torch.load(LOSS_MODULE_PATH))
vgg5.cuda().eval()


################# GPU  #################
cap = cv2.VideoCapture('data/videos/tram_mobile.avi')   #assume it's 576x1024 (HxW)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('data/videos/out_vid.avi', fourcc, 20.0, (1024,576))


criterion = LossCriterion(style_layers = ['r11','r21','r31', 'r41'],
                          content_layers=['r41'],
                          style_weight=0.02,
                          content_weight=0.2)


################## STYLE ####################3



style = cv2.imread(STYLE_PATH)
if RGB:
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
style = style.transpose((2,0,1))
style = torch.from_numpy(style).unsqueeze(0).cuda()
style = style / 255.0
with torch.no_grad():
    sF = e3s(style)
    sF_loss = vgg5(style)

i = 0
tt = time()
with torch.no_grad():
    n = 0
    while(True):
        ret, frame = cap.read()
        if not ret: break
        if RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose((2,0,1))
        frame = torch.from_numpy(frame).unsqueeze(0)
        content = frame.cuda()
        content = content/255.0
        torch.cuda.synchronize()
        T = time()
        transfer = e3c(content)
        transfer = mat3(transfer, sF, n)
        transfer = d3(transfer)
        torch.cuda.synchronize()
        print(time()-T)
        n += 1

        #torch.cuda.synchronize()
        #xx = time()
        #transfer = enc(content)
        #transfer = mat(transfer, sF)
        #transfer = dec(transfer)
        #torch.cuda.synchronize()
        #print('test model', time() - xx, transfer.shape)
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
            transfer = cv2.cvtColor(transfer, cv2.COLOR_RGB2BGR)


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
