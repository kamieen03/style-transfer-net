from libs.models import encoder3, decoder3
from libs.Matrix import MulLayer
import torch.nn


class Transfer3(torch.nn.Module):
    def __init__(self, WIDTH):
        super(Transfer3, self).__init__()
        self.vgg_c = encoder3(WIDTH)
        self.vgg_s = encoder3(WIDTH)
        self.matrix = MulLayer(layer='r31', WIDTH)
        self.dec = decoder3(WIDTH)

    def forward(self, inp): 
        # assuming inp is 5D tensor (B,6,H,W)
        cF = self.vgg_c(inp[:,:3,:,:])
        sF = self.vgg_s(inp[:,3:,:,:])
        cF = self.matrix(cF, sF)
        cF = self.dec(cF)
        return cF

