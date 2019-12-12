from libs.parametric_models import encoder3, decoder3
import torch.nn

class Autoencoder(torch.nn.Module):
    def __init__(self, W):
        super(Autoencoder, self).__init__()
        self.encoder = encoder3(W)
        self.decoder = decoder3(W)

    def forward(self, inp): 
        cF = self.encoder(inp)
        cF = self.decoder(cF)
        return cF

