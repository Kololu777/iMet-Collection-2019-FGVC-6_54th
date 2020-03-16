import torch
import torch.nn as nn
import torch.nn.functional as F


from .DenseNet import densenet121
##########################################################################
# https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109#latest-507888
# model:DenseNet121+head


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class MaxPool(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, x.shape[2:])


class DenseNet121(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = densenet121(pretrained=pre)
        self.linear1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.avg_pool = AvgPool()
        self.max_pool = MaxPool()
        self.linear2 = nn.Linear(1024, 1103)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.BN1 = nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)
        self.BN2 = nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)
    
    def forward(self, x):
        # features equal to convolution~Dense block
        x = self.relu(self.encoder.features(x))
        x1 = self.avg_pool(x).view(x.size(0), -1)  
        x2 = self.max_pool(x).view(x.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.BN1(x) 
        x = self.dropout(x) 
        x = self.relu(self.linear1(x))  
        x = self.BN2(x)  
        x = self.dropout(x)  
        x = self.linear2(x)
    
        return x
