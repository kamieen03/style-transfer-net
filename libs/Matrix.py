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
        b,c,h,w = out.size()
        #print(1, b,c,h,w)
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
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

    def forward(self, cF,sF,alpha=1.0,trans=True):
        #cFBK = cF.clone()
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)
        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cF = cF - cMean

        #sF = alpha*sF + (1-alpha)*cF
        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanS = sMean.expand_as(sF)
        sMeanC = sMean.expand_as(cF)
        sF = sF - sMeanS


        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)

        sMatrix = self.snet(sF)
        cMatrix = self.cnet(cF)

        sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
        cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
        transmatrix = torch.bmm(sMatrix,cMatrix)
        transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
        out = self.unzip(transfeature.view(b,c,h,w))
        out = out + sMeanC
        return out
