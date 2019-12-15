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
from libs.models import encoder5


WIDTH = 0.5
if PARAMETRIC:
    VGG_C_PATH  = f'models/pruned/vgg_c_r31.pth'
    VGG_S_PATH  = f'models/pruned/vgg_s_r31.pth'
    DEC_PATH    = f'models/pruned/dec_r31.pth'
    MATRIX_PATH = f'models/pruned/matrix_r31.pth'
else:
    VGG_C_PATH   = 'models/regular/vgg_r31.pth'
    VGG_S_PATH   = 'models/regular/vgg_r31.pth'
    DEC_PATH     = 'models/regular/dec_r31.pth'
    MATRIX_PATH  = 'models/regular/r31.pth'
LOSS_MODULE_PATH = 'models/regular/vgg_r51.pth'

STYLE_PATH  = 'data/style/27.jpg'
LAYER = 'r31'

################# MODEL #################
if PARAMETRIC:
    matrix = MulLayer('r31', WIDTH)
    vgg_c = encoder3(WIDTH)
    vgg_s = encoder3(WIDTH)
    dec = decoder3(WIDTH)
else:
    matrix = MulLayer('r31')
    vgg_c = encoder3()
    vgg_s = encoder3()
    dec = decoder3()
vgg5 = encoder5()

vgg_c.load_state_dict(torch.load(VGG_C_PATH))
vgg_s.load_state_dict(torch.load(VGG_S_PATH))
dec.load_state_dict(torch.load(DEC_PATH))
matrix.load_state_dict(torch.load(MATRIX_PATH))
vgg5.load_state_dict(torch.load(LOSS_MODULE_PATH))

for param in vgg_c.parameters():
    param.requires_grad = False
for param in vgg_s.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False
for param in vgg5.parameters():
    param.requires_grad = False

################# GLOBAL VARIABLE #################
content = torch.Tensor(1,3,576,1024)
style = torch.Tensor(1,3,256,256)

################# GPU  #################
vgg_c.cuda().eval()
vgg_s.cuda().eval()
dec.cuda().eval()
matrix.cuda().eval()
vgg5.cuda().eval()

style = style.cuda()
content = content.cuda()

#summary(vgg, (3,576,1024))
#summary(matrix, [(256,144,256), (256, 64, 64)])
#summary(dec, (256,144,256))

cap = cv2.VideoCapture('data/videos/tram.avi')   #assume it's 576x1024 (HxW)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('data/videos/out_vid.avi', fourcc, 20.0, (1024,576))

style_tmp = cv2.imread(STYLE_PATH)
#style_tmp = cv2.cvtColor(style_tmp, cv2.COLOR_BGR2RGB)
style_tmp = style_tmp.transpose((2,0,1))
style_tmp = torch.from_numpy(style_tmp).unsqueeze(0)
style.data.copy_(style_tmp)
style = style / 255.0
with torch.no_grad():
    sF = vgg_s(style)
    sF_loss = vgg5(style)

criterion = LossCriterion(style_layers = ['r11','r21','r31', 'r41'],
                          content_layers=['r41'],
                          style_weight=0.02,
                          content_weight=1.0)

i = 0
tt = time()
while(True):
    ret, frame = cap.read()
    if not ret: break
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame).unsqueeze(0)
    content.data.copy_(frame)
    content = content/255.0

    with torch.no_grad():
        cF = vgg_c(content)
        #torch.cuda.synchronize()
        feature = matrix(cF,sF)
        transfer = dec(feature)
        #transfer = dec(cF)

        if '-l' in sys.argv:
            cF_loss = vgg5(content)
            tF = vgg5(transfer)
            loss,styleLoss,contentLoss = criterion(tF,sF_loss,cF_loss)
            print(loss.item(), styleLoss.item(), contentLoss.item())
        transfer = transfer.clamp(0,1).squeeze(0)*255
        transfer = transfer.type(torch.uint8).data.cpu().numpy()
        transfer = transfer.transpose((1,2,0))
        #transfer = cv2.cvtColor(transfer, cv2.COLOR_BGR2RGB)


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
