#!/usr/bin/env python3

import cv2
from PIL import Image
import torch
import numpy as np
import argparse

from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import makeVideo
import torch.backends.cudnn as cudnn
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4
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

################# DATA #################
def loadImg(imgPath):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
                transforms.Scale(opt.fineSize),
                transforms.ToTensor()])
    return transform(img)
style = loadImg(opt.style).unsqueeze(0)

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
dec.load_state_dict(torch.load(opt.dec_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))
for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False

################# GLOBAL VARIABLE #################
content = torch.Tensor(1,3,800,600)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()

    style = style.cuda()
    content = content.cuda()

totalTime = 0
imageCounter = 0
contents = []
styles = []
cap = cv2.VideoCapture('/tmp/in_vid.mp4')   #assume it's 800x600 (HxW)
#cap.set(3,600)
#cap.set(4,800)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('/tmp/out_style.avi', fourcc, 20.0, (600,800))

with torch.no_grad():
    sF = vgg(style)

while(True):
    ret, frame = cap.read()
    frame = frame.transpose((2,1,0))
    frame = torch.from_numpy().unsqueeze(0)
    frame = frame/255.0
    content.data.copy_(frame)
    with torch.no_grad():
        cF = vgg(content)
        if(opt.layer == 'r41'):
            feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
        else:
            feature,transmatrix = matrix(cF,sF)
        transfer = dec(feature)
    transfer = transfer.clamp(0,1).squeeze(0).data.cpu().numpy()
    transfer = transfer.transpose((1,2,0))
    #out.write(np.uint8(transfer*255))
    cv2.imshow('frame',transfer)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()
