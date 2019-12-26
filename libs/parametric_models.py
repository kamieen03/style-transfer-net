import torch
import torch.nn as nn

class encoder3(nn.Module):
    def __init__(self, W):
        super(encoder3,self).__init__()   # W - width
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,int(64*W),3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(int(64*W),int(64*W),3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = False)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(int(64*W),int(128*W),3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(int(128*W),int(128*W),3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = False)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(int(128*W),int(256*W),3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56
    def forward(self,x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        return out

class decoder3(nn.Module):
    def __init__(self, W):
        super(decoder3,self).__init__()
        # decoder
        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(int(256*W),int(128*W),3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(int(128*W),int(128*W),3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(int(128*W),int(64*W),3,1,0)
        self.relu9 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(int(64*W),int(64*W),3,1,0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(int(64*W),3,3,1,0)

    def forward(self,x):
        output = {}
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.unpool2(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        return out

class CNN(nn.Module):
    def __init__(self,W,matrixSize=32):
        super(CNN,self).__init__()
            # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(int(256*W),int(256*W),3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(int(256*W),int(256*W),3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(int(256*W),matrixSize,3,1,1))

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
    def __init__(self,W,matrixSize=32):
        super(MulLayer,self).__init__()
        self.snet = CNN(W,matrixSize)
        self.cnet = CNN(W,matrixSize)
        self.matrixSize = matrixSize

        self.compress = nn.Conv2d(int(256*W),matrixSize,1,1,0)
        self.unzip = nn.Conv2d(matrixSize,int(256*W),1,1,0)
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

        if(trans):
            cMatrix = self.cnet(cF)
            sMatrix = self.snet(sF)

            sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
            cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
            transmatrix = torch.bmm(sMatrix,cMatrix)
            transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
            out = self.unzip(transfeature.view(b,c,h,w))
            out = out + sMeanC
            return out
        else:
            out = self.unzip(compress_content.view(b,c,h,w))
            out = out + cMean
            return out

