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
        self.vgg.load_state_dict(torch.load('models/vgg_r31.pth'))
        self.dec.load_state_dict(torch.load('models/dec_r31.pth'))
        self.matrix.load_state_dict(torch.load('models/r31.pth'))


    def forward(self, content, style):
        cF = self.vgg(content)
        if self.sF is None:
            self.sF = self.vgg(style)
        cF = self.matrix(cF, self.sF)
        cF = self.dec(cF)
        return cF


