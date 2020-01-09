from __future__ import division
import os
import cv2
import time
import torch
import scipy.misc
import numpy as np
import scipy.sparse
from PIL import Image
import scipy.sparse.linalg
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

def whiten(cF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2,cF)
    return whiten_cF

def numpy2cv2(cont,style,prop,width,height):
    cont = cont.transpose((1,2,0))
    cont = cont[...,::-1]
    cont = cont * 255
    cont = cv2.resize(cont,(width,height))
    #cv2.resize(iimg,(width,height))
    style = style.transpose((1,2,0))
    style = style[...,::-1]
    style = style * 255
    style = cv2.resize(style,(width,height))

    prop = prop.transpose((1,2,0))
    prop = prop[...,::-1]
    prop = prop * 255
    prop = cv2.resize(prop,(width,height))

    #return np.concatenate((cont,np.concatenate((style,prop),axis=1)),axis=1)
    return prop,cont

def makeVideo(content,style,props,outf):
    print('Stack transferred frames back to video...')
    layers,height,width = content[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(os.path.join(outf,'transfer.avi'),fourcc,10.0,(width,height))
    ori_video = cv2.VideoWriter(os.path.join(outf,'content.avi'),fourcc,10.0,(width,height))
    for j in range(len(content)):
        prop,cont = numpy2cv2(content[j],style,props[j],width,height)
        cv2.imwrite('prop.png',prop)
        cv2.imwrite('content.png',cont)
        # TODO: this is ugly, fix this
        imgj = cv2.imread('prop.png')
        imgc = cv2.imread('content.png')

        video.write(prop)
        ori_video.write(cont)
        # RGB or BRG, yuks
    video.release()
    ori_video.release()
    os.remove('prop.png')
    os.remove('content.png')
    print('Transferred video saved at %s.'%outf)

def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.outf)
    os.makedirs(expr_dir,exist_ok=True)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

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
    print(W)

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
    print(len(styles))
    total_w = (len(styles) + 1) * W + len(styles)*GAP
    result = np.zeros((total_h, total_w, 3))
    print(result.shape)

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



