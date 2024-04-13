import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # Ensure this is uniquely named and not set elsewhere

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, n_classes, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(12, ninp)  # Assuming input feature size of 12
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, n_classes)  # Decoding to n_classes

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src)  # Encode input features to embeddings
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)  # Apply positional encoding
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = output.mean(dim=0)  # Pool over the sequence length for sequence classification
        return output

# Helper function to instantiate the model
def get_transformer_model(n_classes, ninp, nhead, nhid, nlayers, dropout):
    ntoken = ninp  # the size of vocabulary (not actually used since we're encoding directly to embeddings)
    model = TransformerModel(ntoken, ninp, nhead, nhid, nlayers, n_classes, dropout)
    return model

####### CUSTOM ECG MODEL #######
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # building block for neural network


# class ResidualUnit(nn.Module):
#     def __init__(self, n_filters_in, n_filters_out, kernel_size=17, stride=1, dropout=0.2, activation_function=nn.ReLU):
#         super(ResidualUnit, self).__init__()
#         # first convolutional layer for batch normalization and activation function
#         self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size,
#                                stride=stride, padding=kernel_size//2, bias=False)
#         self.bn1 = nn.BatchNorm1d(n_filters_out)
#         self.activation = activation_function()

#         # second convolutional layer with batch normalization and dropout
#         self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
#                                stride=stride, padding=kernel_size//2, bias=False)
#         self.bn2 = nn.BatchNorm1d(n_filters_out)
#         self.dropout = nn.Dropout(dropout)

#         # check if downsampling is needed
#         self.downsample = n_filters_in != n_filters_out or stride != 1
#         if self.downsample:
#             self.conv_downsample = nn.Conv1d(
#                 n_filters_in, n_filters_out, 1, stride=stride, bias=False)
#             self.bn_downsample = nn.BatchNorm1d(n_filters_out)

#     def forward(self, x):
#         residual = x

#         # first convolutional, batch, activation
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.activation(out)

#         # second convolutional, batch
#         out = self.conv2(out)
#         out = self.bn2(out)

#         # apply downsampling if needed
#         if self.downsample:
#             residual = self.conv_downsample(residual)
#             residual = self.bn_downsample(residual)

#         out += residual
#         out = self.activation(out)
#         out = self.dropout(out)

#         return out

# # define main neural network model for ECG signal analysis


# class ECGNet(nn.Module):
#     def __init__(self, n_classes):
#         super(ECGNet, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(4096, 64, kernel_size=16,
#                       stride=1, padding=8, bias=False),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         )

#         self.residual1 = ResidualUnit(64, 128, dropout=0.2)
#         self.residual2 = ResidualUnit(128, 196, dropout=0.2)
#         self.residual3 = ResidualUnit(196, 256, dropout=0.2)
#         self.residual4 = ResidualUnit(256, 320, dropout=0.2)

#         # flatten layer before converting to fully connected layer
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(2240, n_classes)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.residual1(x)
#         x = self.residual2(x)
#         x = self.residual3(x)
#         x = self.residual4(x)
#         x = self.flatten(x)

#         # debug
#         # print(x.size())

#         x = self.fc(x)

#         return x

# # helper function to instantiate ECGNet model


# def get_model(n_classes):
#     model = ECGNet(n_classes)
#     return model
