#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import argparse
import tensorrt as trt
import sys
import os

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from libs.models import encoder5

PARAMETRIC = '-p' in sys.argv
V2 = 'v2' in sys.argv

WIDTH = 0.25
################# MODEL #################
if PARAMETRIC:
    from libs.parametric_models import encoder3, decoder3, MulLayer
    e3c = encoder3(WIDTH, V2).eval().cuda()
    e3s = encoder3(WIDTH, V2).eval().cuda()
    d3 = decoder3(WIDTH, V2).eval().cuda()
    mat3 = MulLayer(WIDTH).eval().cuda()
    if not V2:
        e3c.load_state_dict(torch.load('models/pruned/vgg_c_r31.pth'))
        e3s.load_state_dict(torch.load('models/pruned/vgg_s_r31.pth'))
        d3.load_state_dict(torch.load('models/pruned/dec_r31.pth'))
        mat3.load_state_dict(torch.load('models/pruned/matrix_r31.pth'))
    else:
        e3c.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/vgg_c_r31.pth'))
        e3s.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/vgg_s_r31.pth'))
        d3.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/dec_r31.pth'))
        mat3.load_state_dict(torch.load('models/prunedv2/prunedv2_0.02_0.6/matrix_r31.pth'))

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



def mosaic(contents, styles, stylized):
    """
    Concatenates, content images, style images and stylized images
    into one big mosaic picture. It is assumed all the stylized images have the same width.
    :param contents: dictionary of content images and their names
    :type  contents: List[numpy.ndarray]
    :param styles: dictionary of style images and theit names
    :type  styles: List[numpy.ndarray]
    :param stylized: dictionary of stylized(result) images and pair
                     of corresponding content and style image names
    :type  stylized: List[List[numpy.ndarray]]
    :returns: mosaic of pictures ordered similarly to e.g.
        https://github.com/FalongShen/styletransfer/raw/master/python/1.png
    :rtype: numpy.ndarray
    """
    GAP = 2     #gap in pixels between images in grid
    _, W, _ = stylized[0][0].shape

    #reshape all pictures to the same width
    for i in range(len(contents)):
        h, w, c = contents[i].shape
        contents[i] = cv2.resize(contents[i], (W, int(W * h/w)))
    for i in range(len(styles)):
        h, w, c = styles[i].shape
        styles[i] = cv2.resize(styles[i], (W, int(W * h/w)))

    #create empty canvas
    max_style_h = int(max([pic.shape[0] for pic in styles]))
    total_h = max_style_h + sum([pic.shape[0] for pic in contents]) + len(contents)*GAP
    total_w = (len(styles) + 1) * W + len(styles)*GAP
    result = np.zeros((total_h, total_w, 3))

    #paint content and style images in the first column and row
    h = max_style_h + GAP
    for pic in contents:
        pic_h, _, _ = pic.shape
        result[h:h + pic_h, :W, :] = pic
        h += pic_h + GAP
    w = W + GAP
    for pic in styles:
        pic_h, _, _ = pic.shape
        result[max_style_h - pic_h:max_style_h, w:w+W, :] = pic
        w += W + GAP
    
    #paint stylized images in correct columns and rows
    h, w = max_style_h + GAP, W + GAP
    for col in stylized:
        for pic in col:
            pic_h, pic_w, _ = pic.shape
            pic = cv2.resize(pic, (W, int(W * pic_h/pic_w)))
            pic_h, _, _ = pic.shape
            result[h:h+pic_h, w:w+W, :] = pic
            h += pic_h + GAP
        w += W + GAP
        h = max_style_h + GAP

    return result





content_list = ['data/content/' + x for x in os.listdir('data/content') if x[-3:] in ['png', 'jpg']]
style_list   = ['data/style/' + x   for x in os.listdir('data/style')   if x[-3:] in ['png', 'jpg']]
contents, styles = [], []

for x in content_list:
    temp = cv2.imread(x)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    contents.append(temp)
for x in style_list:
    temp = cv2.imread(x)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    styles.append(temp)

stylized = []
n = 0
N = len(contents) * len(styles)
i = 1
with torch.no_grad():
    for s_name, s in zip(style_list, styles):
        sF = e3s(torch.from_numpy(s.transpose(2,0,1)).float().unsqueeze(0).cuda()/255.0)
        stylized.append([])
        for c_name, c in zip(content_list, contents):
            print(f'[{i}/{N}] Stylizing {c_name} with {s_name}')
            transfer = e3c(torch.from_numpy(c.transpose(2,0,1)).float().unsqueeze(0).cuda()/255.0)
            transfer = mat3(transfer, sF, n)
            transfer = d3(transfer)
            transfer = transfer.clamp(0,1).squeeze(0)*255
            transfer = transfer.type(torch.uint8).data.cpu().numpy().transpose(1, 2, 0)
            stylized[-1].append(transfer)
            i += 1
            
print("Creating mosaic...")
result = mosaic(contents, styles, stylized)
result = result.astype(np.uint8)
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite('mosaicv1.png', result)


