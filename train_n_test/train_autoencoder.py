#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.abspath(__file__ + "/../../"))   # just so we can use 'libs'

import torch.utils.data
import torch.optim as optim
from torch import nn
import numpy as np

from libs.Loader import Dataset
from libs.Autoencoder import Autoencoder

BATCH_SIZE = 2
CROP_SIZE = 300
ENCODER_SAVE_PATH = 'models/small/vgg_r31.pth'
DECODER_SAVE_PATH = 'models/small/dec_r31.pth'
EPOCHS = 100
WIDTH = 0.5

class Trainer(object):
    def __init__(self):
        datapath = '../data/'

        # set up datasets
        self.train_set = self.load_dataset(datapath+'mscoco/train/')
        self.valid_set = self.load_dataset(datapath+'mscoco/validate/')

        # set up model and loss network
        self.model = Autoencoder(WIDTH)
        self.model.train()
        self.model.cuda()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def load_dataset(self, path):
        """Load the datasets"""
        dataset = Dataset(path, CROP_SIZE)
        loader = torch.utils.data.DataLoader(dataset     = dataset,
                                             batch_size  = BATCH_SIZE,
                                             shuffle     = True,
                                             num_workers = 8,
                                             drop_last   = True)
        return loader

    def train(self):
        best_val = 1e9
        for epoch in range(1, EPOCHS+1): # count from one
            self.train_single_epoch(epoch)
            val = self.validate_single_epoch(epoch)
            if val < best_val:
                best_val = val
                torch.save(self.model.encoder.state_dict(), DECODER_SAVE_PATH)
                torch.save(self.model.decoder.state_dict(), ENCODER_SAVE_PATH)


    def train_single_epoch(self, epoch):
        batch_num = len(self.train_set)      # number of batches in training epoch
        self.model.train()

        for batch_i, content in enumerate(self.train_set):
            content = content[0].cuda() 

            self.optimizer.zero_grad()
            out = self.model(content)

            loss = self.criterion(out, content)
            loss.backward()
            self.optimizer.step()
            print(f'Train Epoch: [{epoch}/{EPOCHS}] ' + 
                  f'Batch: [{batch_i+1}/{batch_num}] ' +
                  f'Loss: {loss:.6f}')

    def validate_single_epoch(self, epoch):
        batch_num = len(self.valid_set)      # number of batches in training epoch
        self.model.eval()
        losses = []
        for batch_i, content in enumerate(self.valid_set):
            content = content[0].cuda()
            out = self.model(content)
            loss = self.criterion(content, out)
            losses.append(loss.item())
            print(f'Validate Epoch: [{epoch}/{EPOCHS}] ' + 
                  f'Batch: [{batch_i+1}/{batch_num}] ' +
                  f'Loss: {loss:.6f}')
        return np.mean(np.array(losses))


def main():
    c = Trainer()
    c.train()

if __name__ == '__main__':
    main()