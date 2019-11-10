from libs.Matrix import MulLayer
from libs.models import encoder3
from libs.models import decoder3
import torch.nn


class Transfer_r3(torch.nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        self.vgg = encoder3()
        self.matrix = MulLayer(layer='r31')
        self.dec = decoder3()
        self.sF = None


    def forward(self, content, style=None): 
        cF = self.vgg(content)
        if style is not None:
            self.sF = self.vgg(style)
        cF = self.matrix(cF, self.sF)
        cF = self.dec(cF)
        return cF

