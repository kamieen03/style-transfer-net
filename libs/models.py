import torch
import torch.nn as nn

class encoder3(nn.Module):
    def __init__(self):
        super(encoder3,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        #self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad1 = nn.ZeroPad2d(1)
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        #self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = False)
        # 112 x 112

        #self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        #self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = False)
        # 56 x 56

        #self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad6 = nn.ZeroPad2d(1)
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56
    def forward(self,x):
        out = self.conv1(x)
        out = self.zeroPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.zeroPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out = self.maxPool(pool1)
        out = self.zeroPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.zeroPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out = self.maxPool2(pool2)
        out = self.zeroPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        return out

class decoder3(nn.Module):
    def __init__(self):
        super(decoder3,self).__init__()
        # decoder
        #self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad7 = nn.ZeroPad2d(1)
        self.conv7 = nn.Conv2d(256,128,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        #self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad8 = nn.ZeroPad2d(1)
        self.conv8 = nn.Conv2d(128,128,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 112 x 112

        #self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad9 = nn.ZeroPad2d(1)
        self.conv9 = nn.Conv2d(128,64,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        #self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad10 = nn.ZeroPad2d(1)
        self.conv10 = nn.Conv2d(64,64,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)

        #self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.zeroPad11 = nn.ZeroPad2d(1)
        self.conv11 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        output = {}
        out = self.zeroPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.zeroPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.zeroPad9(out)
        out = self.conv9(out)
        out_relu9 = self.relu9(out)
        out = self.unpool2(out_relu9)
        out = self.zeroPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.zeroPad11(out)
        out = self.conv11(out)
        return out

class encoder4(nn.Module):
    def __init__(self):
        super(encoder4,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,256,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(256,256,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(256,256,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(256,512,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28
    def forward(self,x,sF=None,matrix11=None,matrix21=None,matrix31=None):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad7(out)

        out = self.conv3(out)
        out = self.relu3(out)

        out = self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad7(out)

        out = self.conv5(out)
        out = self.relu5(out)

        out = self.maxPool2(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)

        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)

        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)

        out = self.maxPool3(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        return out

class decoder4(nn.Module):
    def __init__(self):
        super(decoder4,self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,256,3,1,0)
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(256,256,3,1,0)
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(256,256,3,1,0)
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(256,256,3,1,0)
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(256,128,3,1,0)
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(128,128,3,1,0)
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(128,64,3,1,0)
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(64,64,3,1,0)
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)

        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out

class decoder4(nn.Module):
    def __init__(self):
        super(decoder4,self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,256,3,1,0)
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(256,256,3,1,0)
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(256,256,3,1,0)
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(256,256,3,1,0)
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(256,128,3,1,0)
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(128,128,3,1,0)
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(128,64,3,1,0)
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(64,64,3,1,0)
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)

        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out

class encoder5(nn.Module):
    def __init__(self):
        super(encoder5,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,256,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(256,256,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(256,256,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(256,512,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,512,3,1,0)
        self.relu11 = nn.ReLU(inplace=True)

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(512,512,3,1,0)
        self.relu12 = nn.ReLU(inplace=True)

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(512,512,3,1,0)
        self.relu13 = nn.ReLU(inplace=True)

        self.maxPool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(512,512,3,1,0)
        self.relu14 = nn.ReLU(inplace=True)

    def forward(self,x,sF=None,contentV256=None,styleV256=None,matrix11=None,matrix21=None,matrix31=None):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad7(out)

        out = self.conv3(out)
        out = self.relu3(out)

        out= self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad7(out)

        out = self.conv5(out)
        out = self.relu5(out)

        out= self.maxPool2(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)

        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)

        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)

        out = self.maxPool3(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)

        out = self.reflecPad11(out)
        out = self.conv11(out)
        out = self.relu11(out)

        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)

        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)

        out= self.maxPool4(out)

        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        return out

class decoder5(nn.Module):
    def __init__(self):
        super(decoder5,self).__init__()

        # decoder
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(512,512,3,1,0)
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(256,256,3,1,0)
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        self.relu24 = nn.ReLU(inplace=True)

        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(128,64,3,1,0)
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        self.relu26 = nn.ReLU(inplace=True)

        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        # decoder
        out = self.reflecPad15(x)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out
