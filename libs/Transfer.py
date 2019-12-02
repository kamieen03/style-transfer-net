from libs.Matrix import MulLayer
from libs.models import encoder3, encoder4
from libs.models import decoder3, decoder4
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

class Transfer4(torch.nn.Module):
    def __init__(self):
        super(Transfer4, self).__init__()
        self.vgg = encoder4()
        self.matrix = MulLayer(layer='r41')
        self.dec = decoder4()
        self.sF = None


    def forward(self, content, style=None): 
        cF = self.vgg(content)
        if style is not None:
            self.sF = self.vgg(style)
        cF = self.matrix(cF, self.sF)
        cF = self.dec(cF)
        return cF

