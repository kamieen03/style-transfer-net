#!/usr/bin/env python3

from libs.Transfer import Transfer
import torch

model = Transfer()
torch.save(model.state_dict(), 'models/transfer_r31.pth')
