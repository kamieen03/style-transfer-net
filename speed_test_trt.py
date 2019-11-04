#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import time

from torch2trt import TRTModule
LAYER = 'r31'

################# PREPARATIONS #################

matrix, vgg, dec = TRTModule(), TRTModule(), TRTModule()
if(LAYER == 'r31'):
    matrix.load_state_dict(torch.load('/home/kamil/Desktop/style-transfer-net/models/tensorrt/mul_r31_trt.pth')) 
    vgg.load_state_dict(torch.load('models/tensorrt/vgg_r31_trt.pth')) 
    dec.load_state_dict(torch.load('models/tensorrt/dec_r31_trt.pth')) 
elif(LAYER == 'r41'):
    matrix.load_state_dict(torch.load('models/tensorrt/mul_r41_trt.pth')) 
    vgg.load_state_dict(torch.load('models/tensorrt/vgg_r41_trt.pth')) 
    dec.load_state_dict(torch.load('models/tensorrt/dec_r41_trt.pth')) 

for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False

################# GLOBAL VARIABLE #################
content = torch.Tensor(1,3,1920,1080)
style = torch.Tensor(1,3,1920,1080)

################# GPU  #################
vgg.cuda()
dec.cuda()
matrix.cuda()

style = style.cuda()
content = content.cuda()

################# I/O ##################
cap = cv2.VideoCapture('/home/kamil/Desktop/style-transfer-net/data/videos/in/in_vid.mp4')   #assume it's 800x600 (HxW)
#cap.set(3,600)
#cap.set(4,800)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('/home/kamil/Desktop/style-transfer-net/data/videos/out/out_vid.avi', fourcc, 20.0, (600,800))

style_tmp = cv2.imread('/home/kamil/Desktop/style-transfer-net/data/style/style1.jpg')
style_tmp = style_tmp.transpose((2,0,1))
style_tmp = torch.from_numpy(style_tmp).unsqueeze(0)
style.data.copy_(style_tmp)
style = style / 255.0
with torch.no_grad():
    sF = vgg(style)


############# main #####################
i = 0
tt = time.time()
while(True):
    ret, frame = cap.read()
    if not ret: break
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame).unsqueeze(0)
    content.data.copy_(frame)
    content = content/255.0

    with torch.no_grad():
        cF = vgg(content)
        if(LAYER == 'r41'):
            feature = matrix(cF[LAYER], sF[LAYER])
        else:
            feature = matrix((cF, sF))
        transfer = dec(feature)
    transfer = transfer.clamp(0,1).squeeze(0)*255
    transfer = transfer.type(torch.uint8).data.cpu().numpy()
    transfer = transfer.transpose((1,2,0))

    #out.write(np.uint8(transfer*255))
    #cv2.imshow('frame',transfer)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    i += 1
    print((time.time()-tt)/i)

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()
