#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.abspath(__file__ + "/../../"))   # just so we can use 'libs'

import torch.utils.data
import torch.optim as optim
from torch import nn
import numpy as np
import torch

from libs.Loader import Dataset
from libs.shufflenetv2 import ShuffleNetV2AutoEncoder

BATCH_SIZE = 16
CROP_SIZE = 416
ENCODER_PATH      = f'models/regular/shufflenetv2_x1_encoder.pth'
DECODER_SAVE_PATH = f'models/regular/shufflenetv2_x1_decoder.pth'
EPOCHS = 20

class Trainer(object):
    def __init__(self):
        datapath = '../data/'

        # set up datasets
        self.train_set = self.load_dataset(datapath+'mscoco/train/')
        self.valid_set = self.load_dataset(datapath+'mscoco/validate/')

        # set up model
        self.model = ShuffleNetV2AutoEncoder().cuda()
        # load encoder
        self.model.encoder.load_state_dict(torch.load(ENCODER_PATH))
        self.model.encoder.eval()
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        
        # load decoder
        try:
            self.model.decoder.load_state_dict(torch.load(DECODER_SAVE_PATH))
        except:
            print("Decoder weights not found. Proceeding with new ones...")
        self.model.decoder.train()


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
        with open('shufflenetv2_log.txt', 'w+') as f:
            for epoch in range(1, EPOCHS+1): # count from one
                self.train_single_epoch(epoch, f)
                val = self.validate_single_epoch(epoch, f)
                if val < best_val:
                    best_val = val
                    torch.save(self.model.decoder.state_dict(), DECODER_SAVE_PATH)


    def train_single_epoch(self, epoch, f):
        batch_num = len(self.train_set)      # number of batches in training epoch
        self.model.decoder.train()

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
            f.write(f'Train Epoch: [{epoch}/{EPOCHS}] ' + 
                  f'Batch: [{batch_i+1}/{batch_num}] ' +
                  f'Loss: {loss:.6f}\n')

    def validate_single_epoch(self, epoch, f):
        batch_num = len(self.valid_set)      # number of batches in training epoch
        self.model.decoder.eval()
        losses = []
        with torch.no_grad():
            for batch_i, content in enumerate(self.valid_set):
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
        f.write('Mean:', np.mean(np.array(losses)))
        return np.mean(np.array(losses))


def main():
    c = Trainer()
    c.train()

if __name__ == '__main__':
    main()
