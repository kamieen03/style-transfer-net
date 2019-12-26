#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.abspath(__file__ + "/../../"))   # just so we can use 'libs'

import torch
import torch.optim as optim
from torch import nn
import numpy as np

from libs.Loader import Dataset
from libs.Criterion import LossCriterion
from libs.parametric_models import encoder3, MulLayer, decoder3
from libs.models import encoder5

BATCH_SIZE = 8
CROP_SIZE = 300
WIDTH = 0.25
ENCODER_SAVE_PATH = f'models/pruned/autoencoder/vgg_r31.pth'
DECODER_SAVE_PATH = f'models/pruned/autoencoder/dec_r31.pth'
MATRIX_SAVE_PATH  = f'models/parametric/matrix_r31_W{WIDTH}.pth'
LOSS_MODULE_PATH  = 'models/regular/vgg_r51.pth'
EPOCHS = 10

class Trainer(object):
    def __init__(self):
        datapath = '../data/'

        # set up datasets
        self.content_train, self.style_train = self.load_datasets(
            datapath+'mscoco/train', datapath+'wikiart/train')
        self.content_valid, self.style_valid = self.load_datasets(
            datapath+'mscoco/validate', datapath+'wikiart/validate')


        self.matrix = MulLayer(WIDTH)
        self.matrix.train()
        self.matrix.cuda()
        try:
            self.matrix.load_state_dict(torch.load(MATRIX_SAVE_PATH))
        except:
            print("Matrix checkpoint not found. Proceeding with new weights.")

        self.vgg = encoder3(WIDTH)
        self.vgg.eval()
        self.vgg.cuda()
        self.vgg.load_state_dict(torch.load(ENCODER_SAVE_PATH))

        self.dec = decoder3(WIDTH)
        self.dec.eval()
        self.dec.cuda()
        self.dec.load_state_dict(torch.load(DECODER_SAVE_PATH))

        self.loss_module = encoder5()
        self.loss_module.eval()
        self.loss_module.cuda()
        self.loss_module.load_state_dict(torch.load(LOSS_MODULE_PATH))

        for param in self.vgg.parameters():
            param.requires_grad = False
        for param in self.dec.parameters():
            param.requires_grad = False
        for param in self.loss_module.parameters():
            param.requires_grad = False

        # set up loss function and optimizer
        self.criterion = LossCriterion(style_layers = ['r11','r21','r31','r41'],
                                  content_layers=['r41'],
                                  style_weight=0.02,
                                  content_weight=1.0)
        self.optimizer = optim.Adam(self.matrix.parameters(), lr=1e-4)

    def load_datasets(self, content_path, style_path):
        """Load the datasets"""
        content_dataset = Dataset(content_path, CROP_SIZE)
        content_loader = torch.utils.data.DataLoader(dataset     = content_dataset,
                                                     batch_size  = BATCH_SIZE,
                                                     shuffle     = True,
                                                     num_workers = 8,
                                                     drop_last   = True)
        style_dataset = Dataset(style_path, CROP_SIZE)
        style_loader = torch.utils.data.DataLoader(dataset     = style_dataset,
                                                   batch_size  = BATCH_SIZE,
                                                   shuffle     = True,
                                                   num_workers = 8,
                                                   drop_last   = True)
        return content_loader, style_loader

    def train(self):
        best_val = 1e9
        with open('log_matrix3.txt', 'w+') as f:
            for epoch in range(1, EPOCHS+1): # count from one
                self.train_single_epoch(epoch, f)
                val = self.validate_single_epoch(epoch, f)
                if val < best_val:
                    best_val = val
                    torch.save(self.matrix.state_dict(), MATRIX_SAVE_PATH)


    def train_single_epoch(self, epoch, f):
        batch_num = len(self.content_train)      # number of batches in training epoch
        self.matrix.train()

        for batch_i, (content, style) in enumerate(zip(self.content_train, self.style_train)):
            content = content[0].cuda() 
            style   = style[0].cuda()

            self.optimizer.zero_grad()
            sF = self.vgg(style)
            cF = self.vgg(content)
            feature = self.matrix(cF,sF)
            transfer = self.dec(feature)

            sF_loss = self.loss_module(style)
            cF_loss = self.loss_module(content)
            tF_loss = self.loss_module(transfer)
            loss, styleLoss, contentLoss = self.criterion(tF_loss, sF_loss, cF_loss)

            loss.backward()
            self.optimizer.step()
            print(f'Train Epoch: [{epoch}/{EPOCHS}] ' + 
                  f'Batch: [{batch_i+1}/{batch_num}] ' +
                  f'Loss: {loss:.6f} '+
                  f'StyleLoss: {styleLoss:.6f} ' + 
                  f'ContentLoss: {contentLoss:.6f} ')
            f.write(f'Train Epoch: [{epoch}/{EPOCHS}] ' + 
                  f'Batch: [{batch_i+1}/{batch_num}] ' +
                  f'Loss: {loss:.6f}\n')

    def validate_single_epoch(self, epoch, f):
        batch_num = len(self.content_valid)      # number of batches in training epoch
        self.matrix.eval()
        losses = []
        
        with torch.no_grad():
            for batch_i, (content, style) in enumerate(zip(self.content_valid, self.style_valid)):
                content = content[0].cuda() 
                style   = style[0].cuda()

                sF = self.vgg(style)
                cF = self.vgg(content)
                feature = self.matrix(cF,sF)
                transfer = self.dec(feature)

                sF_loss = self.loss_module(style)
                cF_loss = self.loss_module(content)
                tF_loss = self.loss_module(transfer)
                loss, styleLoss, contentLoss = self.criterion(tF_loss, sF_loss, cF_loss)

                losses.append(loss.item())
                print(f'Validate Epoch: [{epoch}/{EPOCHS}] ' + 
                      f'Batch: [{batch_i+1}/{batch_num}] ' +
                      f'Loss: {loss:.6f} '+
                      f'StyleLoss: {styleLoss:.6f} ' + 
                      f'ContentLoss: {contentLoss:.6f} ')
                f.write(f'Validate Epoch: [{epoch}/{EPOCHS}] ' + 
                      f'Batch: [{batch_i+1}/{batch_num}] ' +
                      f'Loss: {loss:.6f}\n')
        return np.mean(np.array(losses))


def main():
    c = Trainer()
    c.train()

if __name__ == '__main__':
    main()

