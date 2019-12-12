#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.abspath(__file__ + "/../../"))   # just so we can use 'libs'

import time
import torch.optim
import torch.utils.data
import torch.optim as optim
import copy
import distiller
import cv2

from libs.Transfer import Transfer3
from libs.models import encoder5
from libs.Loader import Dataset
from libs.Criterion import LossCriterion

BATCH_SIZE = 2 
CROP_SIZE = 256
VGG_SAVE_PATH = 'models/small/vgg_r31.pth'
MATRIX_SAVE_PATH = 'models/small/matrix_r31.pth'
DECODER_SAVE_PATH = 'models/small/dec_r31.pth'
EPOCHS = 30

class Trainer(object):
    def __init__(self):
        datapath = '../data/'

        # set up datasets
        self.content_train, self.style_train = self.load_datasets(
            datapath+'mscoco/train', datapath+'wikiart/train')
        self.content_valid, self.style_valid = self.load_datasets(
            datapath+'mscoco/validate', datapath+'wikiart/validate')

        # set up model and loss network
        self.auto_encoder = EncoderDecoder()
        self.auto_encoder.train()
        self.auto_encoder.cuda()
        self.matrix = MulLayer('r31')
        self.matrix.train()
        self.matrix.cuda()
        self.loss_module = encoder5()
        self.loss_module.eval()
        self.loss_module.cuda()

        # set up loss function and optimizer
        self.criterion = LossCriterion(style_layers = ['r11','r21','r31','r41'],
                                  content_layers=['r41'],
                                  style_weight=0.02,
                                  content_weight=1.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        # set up compression scheduler
        self.compression_scheduler = distiller.file_config(self.model, self.optimizer,
            compression_schedule_path)


    def load_datasets(self, content_path, style_path):
        """Load the datasets"""
        content_dataset = Dataset(content_path, 256)  #300 isnt used anyway
        content_loader = torch.utils.data.DataLoader(dataset     = content_dataset,
                                                     batch_size  = BATCH_SIZE,
                                                     shuffle     = True,
                                                     num_workers = 3,
                                                     drop_last   = True)
        style_dataset = Dataset(style_path, 256) 
        style_loader = torch.utils.data.DataLoader(dataset     = style_dataset,
                                                   batch_size  = BATCH_SIZE,
                                                   shuffle     = True,
                                                   num_workers = 3,
                                                   drop_last   = True)
        return content_loader, style_loader

    def train(self):
        for epoch in range(1, EPOCHS+1): # count from one
            self.compression_scheduler.on_epoch_begin(epoch)
            self.train_single_epoch(epoch)
            self.validate_single_epoch(epoch)
            self.compression_scheduler.on_epoch_end(epoch)
            torch.save(self.model.vgg_c.state_dict(), VGG_C_SAVE_PATH)
            torch.save(self.model.vgg_s.state_dict(), VGG_S_SAVE_PATH)
            torch.save(self.model.matrix.state_dict(), MATRIX_SAVE_PATH)
            torch.save(self.model.dec.state_dict(), DECODER_SAVE_PATH)

    def train_single_epoch(self, epoch):
        batch_num = len(self.content_train)      # number of batches in training epoch
        self.model.train()

        for batch_i, (content, style) in enumerate(zip(self.content_train, self.style_train)):
            self.compression_scheduler.on_minibatch_begin(epoch, batch_i, batch_num, self.optimizer)
            content, style = content[0].cuda(), style[0].cuda()
            inp = torch.cat((content, style),dim=1)

            self.optimizer.zero_grad()
            transfer = self.model(inp)

            sF_loss = self.loss_module(style)
            cF_loss = self.loss_module(content)
            tF = self.loss_module(transfer)
            loss, style_loss, content_loss = self.criterion(tF,sF_loss,cF_loss)
            self.compression_scheduler.before_backward_pass(epoch, batch_i, batch_num, loss, self.optimizer)
            loss.backward()
            self.compression_scheduler.before_parameter_optimization(epoch, batch_i, batch_num, self.optimizer)
            self.optimizer.step()
            self.compression_scheduler.on_minibatch_end(epoch, batch_i, batch_num, self.optimizer)
            print(f'Train Epoch: [{epoch}/{EPOCHS}] ' + 
                  f'Batch: [{batch_i+1}/{batch_num}] ' +
                  f'Loss: {loss:.4f} contentLoss: {10000*content_loss:.4f} styleLoss: {style_loss:.4f}')

    def validate_single_epoch(self, epoch):
        batch_num = len(self.content_valid)      # number of batches in training epoch
        self.model.eval()

        for batch_i, (content, style) in enumerate(zip(self.content_valid, self.style_valid)):
            content, style = content[0].cuda(), style[0].cuda()
            inp = torch.cat((content, style),dim=1)
            transfer = self.model(inp)
            sF_loss = self.loss_module(style)
            cF_loss = self.loss_module(content)
            tF = self.loss_module(transfer)
            loss, style_loss, content_loss = self.criterion(tF,sF_loss,cF_loss)

            print(f'Validate Epoch: [{epoch}/{EPOCHS}] ' + 
                  f'Batch: [{batch_i+1}/{batch_num}] ' +
                  f'Loss: {loss:.4f} contentLoss: {10000*content_loss:.4f} styleLoss: {style_loss:.4f}')


def main():
    c = Compressor()
    c.train()

if __name__ == '__main__':
    main()
