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
import numpy as np

from libs.Transfer import Transfer3
from libs.models import encoder5
from libs.Loader import Dataset
from libs.Criterion import LossCriterion

BATCH_SIZE = 8 
CROP_SIZE = 256
VGG_C_SAVE_PATH = 'models/pruned/vgg_c_r31.pth'
VGG_S_SAVE_PATH = 'models/pruned/vgg_s_r31.pth'
MATRIX_SAVE_PATH = 'models/pruned/matrix_r31.pth'
DECODER_SAVE_PATH = 'models/pruned/dec_r31.pth'
EPOCHS = 21
WIDTH = 0.5

class Compressor(object):
    def __init__(self):
        datapath = '../data/'
        vgg_path = 'models/regular/vgg_r31.pth'
        matrix_path = 'models/regular/r31.pth'
        decoder_path = 'models/regular/dec_r31.pth'
        loss_module_path = 'models/regular/vgg_r51.pth'
        compression_schedule_path = 'prune/schedule.yaml'

        # set up datasets
        self.content_train, self.style_train = self.load_datasets(
            datapath+'mscoco/small', datapath+'wikiart/small')
        self.content_valid, self.style_valid = self.load_datasets(
            datapath+'mscoco/small', datapath+'wikiart/small')

        # set up model and loss network
        self.model = Transfer3()
        self.model.vgg_c.load_state_dict(torch.load(vgg_path))
        self.model.vgg_s.load_state_dict(torch.load(vgg_path))
        self.model.matrix.load_state_dict(torch.load(matrix_path))
        self.model.dec.load_state_dict(torch.load(decoder_path))
        self.model.train()
        self.model.cuda()

        self.loss_module = encoder5()
        self.loss_module.eval()
        self.loss_module.cuda()
        self.loss_module.load_state_dict(torch.load(loss_module_path))

        for param in self.loss_module.parameters():
            param.requires_grad = False


        # set up loss function and optimizer
        self.criterion = LossCriterion(style_layers = ['r11','r21','r31', 'r41'],
                                  content_layers=['r41'],
                                  style_weight=0.02,
                                  content_weight=1.0)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)

        # set up compression scheduler
        self.compression_scheduler = distiller.file_config(self.model, self.optimizer,
            compression_schedule_path)


    def load_datasets(self, content_path, style_path):
        """Load the datasets"""
        content_dataset = Dataset(content_path, CROP_SIZE)
        content_loader = torch.utils.data.DataLoader(dataset     = content_dataset,
                                                     batch_size  = BATCH_SIZE,
                                                     shuffle     = True,
                                                     num_workers = 4,
                                                     drop_last   = True)
        style_dataset = Dataset(style_path, CROP_SIZE) 
        style_loader = torch.utils.data.DataLoader(dataset     = style_dataset,
                                                   batch_size  = BATCH_SIZE,
                                                   shuffle     = True,
                                                   num_workers = 4,
                                                   drop_last   = True)
        return content_loader, style_loader

    def train(self):
        best_val = 1e9
        with open('log_prune.txt', 'w+') as f:
            for epoch in range(1, EPOCHS+1): # count from one
                self.compression_scheduler.on_epoch_begin(epoch)
                self.train_single_epoch(epoch, f)
                val = self.validate_single_epoch(epoch, f)
                self.compression_scheduler.on_epoch_end(epoch)
                if val < best_val:
                    best_val = val
                    torch.save(self.model.vgg_c.state_dict(), VGG_C_SAVE_PATH)
                    torch.save(self.model.vgg_s.state_dict(), VGG_S_SAVE_PATH)
                    torch.save(self.model.matrix.state_dict(), MATRIX_SAVE_PATH)
                    torch.save(self.model.dec.state_dict(), DECODER_SAVE_PATH)

    def train_single_epoch(self, epoch, f):
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
            if loss.item() > 20:
                continue
            self.compression_scheduler.before_backward_pass(epoch, batch_i, batch_num, loss, self.optimizer)
            loss.backward()
            self.compression_scheduler.before_parameter_optimization(epoch, batch_i, batch_num, self.optimizer)
            self.optimizer.step()
            self.compression_scheduler.on_minibatch_end(epoch, batch_i, batch_num, self.optimizer)
            print(f'Train Epoch: [{epoch}/{EPOCHS}] ' + 
                  f'Batch: [{batch_i+1}/{batch_num}] ' +
                  f'Loss: {loss:.6f} contentLoss: {content_loss:.6f} styleLoss: {style_loss:.6f}')
            f.write(f'Train Epoch: [{epoch}/{EPOCHS}] ' + 
                    f'Batch: [{batch_i+1}/{batch_num}] ' +
                    f'Loss: {loss:.6f} contentLoss: {content_loss:.6f} styleLoss: {style_loss:.6f}\n')

    def validate_single_epoch(self, epoch, f):
        batch_num = len(self.content_valid)      # number of batches in training epoch
        self.model.eval()
        self.optimizer.zero_grad()

        losses = []
        with torch.no_grad():
            for batch_i, (content, style) in enumerate(zip(self.content_valid, self.style_valid)):
                content, style = content[0].cuda(), style[0].cuda()
                inp = torch.cat((content, style),dim=1)
                transfer = self.model(inp)
                sF_loss = self.loss_module(style)
                cF_loss = self.loss_module(content)
                tF = self.loss_module(transfer)
                loss, style_loss, content_loss = self.criterion(tF,sF_loss,cF_loss)
                losses.append(loss.item())

                print(f'Validate Epoch: [{epoch}/{EPOCHS}] ' + 
                      f'Batch: [{batch_i+1}/{batch_num}] ' +
                      f'Loss: {loss:.6f} contentLoss: {content_loss:.6f} styleLoss: {style_loss:.6f}')
                f.write(f'Validate Epoch: [{epoch}/{EPOCHS}] ' + 
                        f'Batch: [{batch_i+1}/{batch_num}] ' +
                        f'Loss: {loss:.6f} contentLoss: {content_loss:.6f} styleLoss: {style_loss:.6f}\n')
        f.write(f'mean validation loss: {np.mean(np.array(losses))}')
        return np.mean(np.array(losses))


def main():
    c = Compressor()
    c.train()

if __name__ == '__main__':
    main()
