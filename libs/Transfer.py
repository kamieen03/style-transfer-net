from libs.Matrix import MulLayer
from libs.models import encoder3
from libs.models import decoder3
import torch.nn


class Transfer(torch.nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        self.vgg = encoder3()
        self.matrix = MulLayer(layer='r31')
        self.dec = decoder3()
        self.sF = None


    def forward(self, content_style): 
        # TensorRT has a problem with multiple inputs so we concatenate content and style in one tensor
        # of shape (2,3,H,W); notice content and style have to be of the same shape.
        cF = self.vgg(content_style[0].unsqueeze(0))
        if self.sF is None:
            self.sF = self.vgg(content_style[1].unsqueeze(0))
        cF = self.matrix(cF, self.sF)
        cF = self.dec(cF)
        return cF


