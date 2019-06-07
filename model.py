#pip install EfficientNet
from efficientnet_pytorch import EfficientNet
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import torch.optim as optim

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b3')
        self.linear1 = nn.Linear(1000,256)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(256,2)
        #self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.7)
    def forward(self, input):
        am = self.effnet(input)
        x = self.dropout(self.relu(self.linear1(am)))
        x = self.linear2(x)
        return x
model = classifier()
model.to(device)
criterion = nn.CrossEntropyLoss()

#find best learning rate
from lr_finder import LRFinder
optimizer_ft = optim.Adam(model.parameters(), lr=0.0000001)
lr_finder = LRFinder(model, optimizer_ft, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=1, num_iter=1000)
lr_finder.reset()
lr_finder.plot()
