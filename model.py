import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in "Attention Is All You Need", used in Transformer models.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model's output.
            max_len (int): The maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # This is not a parameter but should be part of the model's state.

    def forward(self, x):
        """
        Adds positional encoding to input sequence.

        Args:
            x (torch.Tensor): Input tensor with shape (sequence length, batch size, model dimension).

        Returns:
            torch.Tensor: Tensor with positional encoding added to input.
        """
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    """
    A transformer model which uses an encoder and positional encoding for sequence processing.
    """
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, n_classes, dropout=0.5):
        """
        Initializes the TransformerModel.

        Args:
            ntoken (int): Number of tokens (size of the vocabulary).
            ninp (int): Number of input features (model dimensionality).
            nhead (int): Number of heads in the multi-head attention models.
            nhid (int): The dimension of the feedforward network model in nn.TransformerEncoder.
            nlayers (int): The number of nn.TransformerEncoderLayer layers.
            n_classes (int): Number of classes for the output layer.
            dropout (float): The dropout value.
        """
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(12, ninp)  # Assumed input feature size is 12
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, n_classes)  # Output layer to predict n_classes

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        """
        Generates a square mask for the sequence. The mask shows which keys have access to which values.

        Args:
            sz (int): Size of the mask (length of the sequence).

        Returns:
            torch.Tensor: A mask tensor indicating allowed positions for self-attention.
        """
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def init_weights(self):
        """
        Initializes weights of the Transformer model with uniform distribution.
        """
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Defines the forward pass of the Transformer model.

        Args:
            src (torch.Tensor): The sequence of embedding tensors.
            src_mask (torch.Tensor): The mask tensor for the sequence.

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        src = self.encoder(src)  # Encode input features to embeddings
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)  # Apply positional encoding
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = output.mean(dim=0)  # Pool over the sequence length for sequence classification
        return output

def get_transformer_model(n_classes, ninp, nhead, nhid, nlayers, dropout):
    """
    Helper function to instantiate a Transformer model with specified parameters.

    Args:
        n_classes (int): Number of target classes.
        ninp (int): Input dimension (embedding size).
        nhead (int): Number of attention heads.
        nhid (int): Dimension of the feedforward network.
        nlayers (int): Number of encoder layers.
        dropout (float): Dropout rate.

    Returns:
        TransformerModel: An instance of the TransformerModel.
    """
    ntoken = ninp  # Assuming vocabulary size matches input dimension (not used since direct encoding)
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
#                                stride=stride, padding=kernel_size//2, bias=False)b
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
