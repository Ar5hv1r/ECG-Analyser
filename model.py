import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualUnit(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, kernel_size=17, stride=1, dropout=0.2, activation_function=nn.ReLU):
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.activation = activation_function()

        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout = nn.Dropout(dropout)

        self.downsample = n_filters_in != n_filters_out or stride != 1
        if self.downsample:
            self.conv_downsample = nn.Conv1d(n_filters_in, n_filters_out, 1, stride=stride, bias=False)
            self.bn_downsample = nn.BatchNorm1d(n_filters_out)
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.conv_downsample(residual)
            residual = self.bn_downsample(residual)

        out += residual
        out = self.activation(out)
        out = self.dropout(out)

        return out

class ECGNet(nn.Module):
    def __init__(self, n_classes):
        super(ECGNet, self).__init__()
        self.layer1 = nn.Sequential (
            nn.Conv1d(12, 64, kernel_size=16, stride=1, padding=8, bias=False), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
        
        self.residual1 = ResidualUnit(64, 128, dropout=0.2)
        self.residual2 = ResidualUnit(128, 196, dropout=0.2)
        self.residual3 = ResidualUnit(196, 256, dropout=0.2)
        self.residual4 = ResidualUnit(256, 320, dropout=0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(320, n_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

def get_model(n_classes):
    model = ECGNet(n_classes)
    return model
