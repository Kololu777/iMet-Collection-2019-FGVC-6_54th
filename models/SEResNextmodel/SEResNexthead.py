import torch.nn.functional as F
import torch
import torch.nn as nn
from .SEResNext import se_resnext50_32x4d, se_resnext101_32x4d


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class MaxPool(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, x.shape[2:])


class GAPseResNext50(nn.Module):
    def __init__(self):
        super(GAPseResNext50, self).__init__()
        self.basemodel = se_resnext50_32x4d()
        self.avg_pool = AvgPool()
        self.max_pool = MaxPool()
        self.linear1 = nn.Linear(3584, 1200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1200, 1103)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.BN1 = nn.BatchNorm1d(3584, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)
        self.BN2 = nn.BatchNorm1d(1200, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)

    def features(self, x):
        batch_size, _, _, _ = x.shape
        x1 = self.basemodel.layer0(x)
        x2 = self.basemodel.layer1(x1)
        x3 = self.basemodel.layer2(x2)
        x4 = self.basemodel.layer3(x3)
        x5 = self.basemodel.layer4(x4)
        x3 = self.avg_pool(x3)
        x4 = self.avg_pool(x4)
        x5 = self.avg_pool(x5)
        output = torch.cat((x3, x4, x5), 1).view(batch_size, -1)
        return output

    def classify(self, x):
        f = self.cls(x)
        return f

    def forward(self, x):
        x = self.features(x)
        x = self.BN1(x)
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.BN2(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GAPseResNext101(nn.Module):
    def __init__(self):
        super(GAPseResNext101, self).__init__()
        self.basemodel = se_resnext101_32x4d()
        self.avg_pool = AvgPool()
        self.max_pool = MaxPool()
        self.linear1 = nn.Linear(3584, 1200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1200, 1103)
        self.dropout = nn.Dropout(0.5)
        self.BN1 = nn.BatchNorm1d(3584, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)
        self.BN2 = nn.BatchNorm1d(1200, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)

    def features(self, x):
        batch_size, _, _, _ = x.shape
        x1 = self.basemodel.layer0(x)
        x2 = self.basemodel.layer1(x1)
        x3 = self.basemodel.layer2(x2)
        x4 = self.basemodel.layer3(x3)
        x5 = self.basemodel.layer4(x4)
        x3 = self.avg_pool(x3)
        x4 = self.avg_pool(x4)
        x5 = self.avg_pool(x5)
        output = torch.cat((x3, x4, x5), 1).view(batch_size, -1)
        return output

    def classify(self, x):
        f = self.cls(x)
        return f

    def forward(self, x):
        x = self.features(x)

        x = self.BN1(x)
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.BN2(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
