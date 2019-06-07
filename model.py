from __future__ import print_function, with_statement, division
import os
os.system('pip install efficientnet_pytorch')
from efficientnet_pytorch import EfficientNet
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import torch.optim as optim
import copy
import os
import torch
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from lr_finder import LRFinder
class classifie(nn.Module):
    def __init__(self,model,n_classes,pretrained = True ):
        super(classifie, self).__init__()
        if(model == 'efficientnet-b3'):
            if(pretrained == True):
                self.cnn_arch = EfficientNet.from_pretrained('efficientnet-b3')
            else:
                self.cnn_arch = EfficientNet.from_name('efficientnet-b3')
        if(model == 'efficientnet-b2'):
            if(pretrained == True):
                self.cnn_arch = EfficientNet.from_pretrained('efficientnet-b2')
            else:
                self.cnn_arch = EfficientNet.from_name('efficientnet-b2')
        if(model == 'efficientnet-b1'):
            if(pretrained == True):
                self.cnn_arch = EfficientNet.from_pretrained('efficientnet-b1')
            else:
                self.cnn_arch = EfficientNet.from_name('efficientnet-b1')
        if(model == 'efficientnet-b0'):
            if(pretrained == True):
                self.cnn_arch = EfficientNet.from_pretrained('efficientnet-b0')
            else:
                self.cnn_arch = EfficientNet.from_name('efficientnet-b0')
        if(model == 'resnet18'):
            self.cnn_arch = models.resnet18(pretrained = pretrained)
        if(model == 'resnet34'):
            self.cnn_arch = models.resnet34(pretrained = pretrained)
        if(model == 'resnet50'):
            self.cnn_arch = models.resnet50(pretrained = pretrained)
        if(model == 'resnet101'):
            self.cnn_arch = models.resnet101(pretrained = pretrained)
        if(model == 'resnet152'):
            self.cnn_arch = models.resnet152(pretrained = pretrained)
        if(model == 'densenet121'):
            self.cnn_arch = models.densenet121(pretrained = pretrained)
        if(model == 'densenet161'):
            self.cnn_arch = models.densenet161(pretrained = pretrained)
        if(model == 'densenet169'):
            self.cnn_arch = models.densenet169(pretrained = pretrained)
        if(model == 'densenet201'):
            self.cnn_arch = models.densenet201(pretrained = pretrained)
        if(model == 'squeezenet1_0'):
            self.cnn_arch = models.squeezenet1_0(pretrained = pretrained)
        if(model == 'squeezenet1_1'):
            self.cnn_arch = models.squeezenet1_1(pretrained = pretrained)
        if(model == 'shufflenet_v2_x0_5'):
            self.cnn_arch = models.shufflenet_v2_x0_5(pretrained = pretrained)
        if(model == 'shufflenet_v2_x1_0'):
            self.cnn_arch = models.shufflenet_v2_x1_0(pretrained = pretrained)
        if(model == 'shufflenet_v2_x1_5'):
            self.cnn_arch = models.shufflenet_v2_x1_5(pretrained = pretrained)
        if(model == 'shufflenet_v2_x2_0'):
            self.cnn_arch = models.shufflenet_v2_x2_0(pretrained = pretrained)
        if(model == 'resnext50_32x4d'):
            self.cnn_arch = models.resnext50_32x4d(pretrained = pretrained)
        if(model == 'resnext101_32x8d'):
            self.cnn_arch = models.resnext101_32x8d(pretrained = pretrained)

        self.linear1 = nn.Linear(1000,256)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(256,n_classes)
        #self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.7)
    def forward(self, input):
        am = self.cnn_arch(input)
        x = self.dropout(self.relu(self.linear1(am)))
        x = self.linear2(x)
        return x
def classifier(model,n_classes,device = 'cpu',pretrained = True):
    model = classifie(model,n_classes,pretrained)
    model.to(device)
    return model
#find best learning rate
def lr_finder(model,train_loader,device):
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0000001)
    lr_finder = LRFinder(model, optimizer_ft, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=1000)
    lr_finder.reset()
    lr_finder.plot()
