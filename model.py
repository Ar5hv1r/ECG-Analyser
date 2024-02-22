import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# building block for neural network


class ResidualUnit(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, kernel_size=17, stride=1, dropout=0.2, activation_function=nn.ReLU):
        super(ResidualUnit, self).__init__()
        # first convolutional layer for batch normalization and activation function
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.activation = activation_function()

        # second convolutional layer with batch normalization and dropout
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout = nn.Dropout(dropout)

        # check if downsampling is needed
        self.downsample = n_filters_in != n_filters_out or stride != 1
        if self.downsample:
            self.conv_downsample = nn.Conv1d(
                n_filters_in, n_filters_out, 1, stride=stride, bias=False)
            self.bn_downsample = nn.BatchNorm1d(n_filters_out)

    def forward(self, x):
        residual = x

        # first convolutional, batch, activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # second convolutional, batch
        out = self.conv2(out)
        out = self.bn2(out)

        # apply downsampling if needed
        if self.downsample:
            residual = self.conv_downsample(residual)
            residual = self.bn_downsample(residual)

        out += residual
        out = self.activation(out)
        out = self.dropout(out)

        return out

# define main neural network model for ECG signal analysis


class ECGNet(nn.Module):
    def __init__(self, n_classes):
        super(ECGNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(4096, 64, kernel_size=16,
                      stride=1, padding=8, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.residual1 = ResidualUnit(64, 128, dropout=0.2)
        self.residual2 = ResidualUnit(128, 196, dropout=0.2)
        self.residual3 = ResidualUnit(196, 256, dropout=0.2)
        self.residual4 = ResidualUnit(256, 320, dropout=0.2)

        # flatten layer before converting to fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2240, n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.flatten(x)

        # debug
        # print(x.size())

        x = self.fc(x)

        return x

# helper function to instantiate ECGNet model


def get_model(n_classes):
    model = ECGNet(n_classes)
    return model
