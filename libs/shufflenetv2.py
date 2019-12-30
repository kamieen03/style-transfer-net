import torch
import torch.nn as nn
from time import time


__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x1_encoder','shufflenet_v2_x1_decoder' ,
    'ShuffleNetV2AutoEncoder', 'MulLayer'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class EncoderResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(EncoderResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class DecoderResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(DecoderResidual, self).__init__()

        self.stride = stride
        branch_features = oup // 2

        if self.stride == 1:
            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, 1, 1, bias=False, groups=branch_features),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            assert branch_features % 2 == 0

            self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
            self.branch = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup, oup, 3, 1, 1, bias=False, groups=oup),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
                    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((self.branch1(x1), self.branch2(x2)), dim=1)
            out = channel_shuffle(out, 2)
        else:
            out = self.unpool(x)
            out = self.branch(out)
        return out

class ShuffleNetV2Encoder(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels,
        num_classes=1000, inverted_residual=EncoderResidual):
        super(ShuffleNetV2Encoder, self).__init__()

        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        stage_names = ['stage{}'.format(i) for i in [1, 2, 3]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        #x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        #x = self.stage4(x)
        #x = self.conv5(x)
        #x = x.mean([2, 3])  # globalpool
        #x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ShuffleNetV2Decoder(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, input_channels, inverted_residual=DecoderResidual):
        super(ShuffleNetV2Decoder, self).__init__()

        self._stage_out_channels = stages_out_channels

        output_channels = self._stage_out_channels[0]

        stage_names = ['stage{}'.format(i) for i in [4, 3, 2]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels):
            seq = []
            for i in range(repeats - 1):
                seq.append(inverted_residual(input_channels, input_channels, 1))
            seq.append(inverted_residual(input_channels, output_channels, 2))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, 1, 1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        x = self.stage4(x)
        x = self.stage3(x)
        x = self.stage2(x)
        x = self.unpool(x)
        x = self.conv2(x)
        return x



def shufflenet_v2_x05(**kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_encoder(**kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ShuffleNetV2Encoder([4, 4, 8], [24, 64, 116, 232], **kwargs)


def shufflenet_v2_x1_decoder():
   return ShuffleNetV2Decoder([8, 4, 4], [116, 24, 16], 232)


class ShuffleNetV2AutoEncoder(nn.Module):
    def __init__(self):
        super(ShuffleNetV2AutoEncoder, self).__init__()
        self.encoder = shufflenet_v2_x1_encoder()
        self.decoder = shufflenet_v2_x1_decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    


class CNN(nn.Module):
    def __init__(self,W,matrixSize=32):
        super(CNN,self).__init__()
            # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(232,int(128*W),3,1,1),
                                   nn.BatchNorm2d(int(128*W)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(int(128*W),int(64*W),3,1,1),
                                   nn.BatchNorm2d(int(64*W)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(int(64*W),matrixSize,3,1,1))

        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        #print(1, b,c,h,w)
        out = out.view(b,c, -1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        #print(2,out.size())
        # 32x32
        out = out.view(b,-1)
        return self.fc(out)

class MulLayer(nn.Module):
    def __init__(self,W=1.0,matrixSize=32):
        super(MulLayer,self).__init__()
        self.snet = CNN(W,matrixSize)
        self.cnet = CNN(W,matrixSize)
        self.matrixSize = matrixSize

        self.compress = nn.Conv2d(232,matrixSize,1,1,0)
        self.unzip = nn.Conv2d(matrixSize,232,1,1,0)
        self.transmatrix = None
        self.W = W

    def forward(self, cF,sF,trans=True):
        cFBK = cF.clone()
        cb, cc, ch, cw = cF.size()
        cFF = cF.view(cb, cc, -1)

        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cF = cF - cMean

        sb, sc, sh, sw = sF.size()
        sFF = sF.view(sb, sc, -1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS

        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)

        cMatrix = self.cnet(cF)
        sMatrix = self.snet(sF)

        sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
        cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
        transmatrix = torch.bmm(sMatrix,cMatrix)
        transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
        out = self.unzip(transfeature.view(b,c,h,w))
        out = out + sMeanC
        return out

