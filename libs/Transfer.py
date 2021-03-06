from libs.models import encoder3, decoder3
from libs.Matrix import MulLayer
import torch.nn


class Transfer3(torch.nn.Module):
    def __init__(self):
        super(Transfer3, self).__init__()
        self.vgg_c = encoder3()
        self.vgg_s = encoder3()
        self.matrix = MulLayer(layer='r31')
        self.dec = decoder3()

    def forward(self, inp): 
        # assuming inp is 5D tensor (B,6,H,W)
        cF = self.vgg_c(inp[:,:3,:,:])
        sF = self.vgg_s(inp[:,3:,:,:])
        cF = self.matrix(cF, sF)
        cF = self.dec(cF)
        return cF

