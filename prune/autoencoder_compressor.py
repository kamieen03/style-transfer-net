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
from torch import nn

from libs.Autoencoder import Autoencoder
from libs.Loader import Dataset

BATCH_SIZE = 16 
CROP_SIZE = 300
VGG_SAVE_PATH = 'models/pruned/autoencoder/vgg_r31.pth'
DECODER_SAVE_PATH = 'models/pruned/autoencoder/dec_r31.pth'
EPOCHS = 11
WIDTH = 0.5

class Compressor(object):
    def __init__(self):
        datapath = '../data/'
        vgg_path = 'models/regular/vgg_r31.pth'
        decoder_path = 'models/regular/dec_r31.pth'
        compression_schedule_path = 'prune/autoencoder_schedule.yaml'

        # set up datasets
        self.content_train = self.load_dataset(
            datapath+'mscoco/prune_train')
        self.content_valid = self.load_dataset(
            datapath+'mscoco/validate')

        # set up model and loss network
        self.model = Autoencoder(1)
        self.model.encoder.load_state_dict(torch.load(vgg_path))
        self.model.decoder.load_state_dict(torch.load(decoder_path))
        self.model.train()
        self.model.cuda()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4)

        # set up compression scheduler
        self.compression_scheduler = distiller.file_config(self.model, self.optimizer,
            compression_schedule_path)


    def load_dataset(self, content_path):
        """Load the datasets"""
        content_dataset = Dataset(content_path, CROP_SIZE)
        content_loader = torch.utils.data.DataLoader(dataset     = content_dataset,
                                                     batch_size  = BATCH_SIZE,
                                                     shuffle     = True,
                                                     num_workers = 4,
                                                     drop_last   = True)
        return content_loader

    def train(self):
        best_val = 1e9
        with open('log_autoencoder_prune.txt', 'w+') as f:
            for epoch in range(1, EPOCHS+1): # count from one
                self.compression_scheduler.on_epoch_begin(epoch)
                self.train_single_epoch(epoch, f)
                #val = self.validate_single_epoch(epoch, f)
                self.compression_scheduler.on_epoch_end(epoch)
                torch.save(self.model.encoder.state_dict(), VGG_SAVE_PATH)
                torch.save(self.model.decoder.state_dict(), DECODER_SAVE_PATH)

    def train_single_epoch(self, epoch, f):
        batch_num = len(self.content_train)      # number of batches in training epoch
        self.model.train()

        for batch_i, content in enumerate(self.content_train):
            self.compression_scheduler.on_minibatch_begin(epoch, batch_i, batch_num, self.optimizer)
            content = content[0].cuda() 

            self.optimizer.zero_grad()
            out = self.model(content)

            loss = self.criterion(content, out)
            if loss.item() > 20:
                continue
            self.compression_scheduler.before_backward_pass(epoch, batch_i, batch_num, loss, self.optimizer)
            loss.backward()
            self.compression_scheduler.before_parameter_optimization(epoch, batch_i, batch_num, self.optimizer)
            self.optimizer.step()
            self.compression_scheduler.on_minibatch_end(epoch, batch_i, batch_num, self.optimizer)
            print(f'Train Epoch: [{epoch}/{EPOCHS}] ' + 
                  f'Batch: [{batch_i+1}/{batch_num}] ' +
                  f'Loss: {loss:.6f}')
            f.write(f'Train Epoch: [{epoch}/{EPOCHS}] ' + 
                    f'Batch: [{batch_i+1}/{batch_num}] ' +
                    f'Loss: {loss:.6f}\n')

    def validate_single_epoch(self, epoch, f):
        batch_num = len(self.content_valid)      # number of batches in training epoch
        self.model.eval()
        self.optimizer.zero_grad()

        losses = []
        with torch.no_grad():
            for batch_i, content in enumerate(self.content_valid):
                content = content[0].cuda()
                out = self.model(content)
                loss = self.criterion(content, out)
                losses.append(loss.item())

                print(f'Validate Epoch: [{epoch}/{EPOCHS}] ' + 
                      f'Batch: [{batch_i+1}/{batch_num}] ' +
                      f'Loss: {loss:.6f}')
                f.write(f'Validate Epoch: [{epoch}/{EPOCHS}] ' + 
                        f'Batch: [{batch_i+1}/{batch_num}] ' +
                        f'Loss: {loss:.6f}\n')
        f.write(f'mean validation loss: {np.mean(np.array(losses))}\n')
        return np.mean(np.array(losses))


def main():
    c = Compressor()
    c.train()

if __name__ == '__main__':
    main()
