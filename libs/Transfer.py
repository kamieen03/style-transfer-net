from libs.Matrix import MulLayer
from libs.models import encoder3, encoder4
from libs.models import decoder3, decoder4
import torch.nn


class Transfer3(torch.nn.Module):
    def __init__(self):
        super(Transfer3, self).__init__()
        self.vgg = encoder3()
        self.matrix = MulLayer(layer='r31')
        self.dec = decoder3()

    def forward(self, inp): 
        # assuming inp is 5D tensor (2,B,C,H,W)
        cF = self.vgg(inp[0])
        sF = self.vgg(inp[1])
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

