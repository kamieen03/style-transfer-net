import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(CNN,self).__init__()
        if(layer == 'r31'):
            # 256x64x64
            self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128,64,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64,matrixSize,3,1,1))
        elif(layer == 'r41'):
            # 512x32x32
            self.convs = nn.Sequential(nn.Conv2d(512,256,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128,matrixSize,3,1,1))

        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        #b,c,h,w = out.size()
        #print(1, b,c,h,w)
        out = out.view(1,32,144*256)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(144*256)
        #print(2,out.size())
        # 32x32
        out = out.view(1,-1)
        return self.fc(out)

class MulLayer(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(MulLayer,self).__init__()
        self.snet = CNN(layer,matrixSize)
        self.cnet = CNN(layer,matrixSize)
        self.matrixSize = matrixSize

        if(layer == 'r41'):
            self.compress = nn.Conv2d(512,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,512,1,1,0)
        elif(layer == 'r31'):
            self.compress = nn.Conv2d(256,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,256,1,1,0)
        self.transmatrix = None

    def forward(self, cF,sF,trans=True):
        cFBK = cF.clone()
        cFF = cF.view(1,256, 144*256)

        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cF = cF - cMean

        sFF = sF.view(1,256,144*256)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sF = sF - sMean


        compress_content = self.compress(cF)
        compress_content = compress_content.view(1,32,-1)

        if(trans):
            cMatrix = self.cnet(cF)
            sMatrix = self.snet(sF)

            sMatrix = sMatrix.view(1,self.matrixSize,self.matrixSize) #1-batchSIZE
            cMatrix = cMatrix.view(1,self.matrixSize,self.matrixSize)
            transmatrix = torch.bmm(sMatrix,cMatrix)
            transfeature = torch.bmm(transmatrix,compress_content).view(1,32,144,256)
            out = self.unzip(transfeature.view(1,32,144,256))
            out = out + sMean
            return out
        else:
            out = self.unzip(compress_content.view(1,32,144,256))
            out = out + cMean
            return out
