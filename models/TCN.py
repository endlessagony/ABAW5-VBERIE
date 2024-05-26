import torch
import torch.nn as nn
from   torch.nn.utils import weight_norm
from   torch.nn.init import xavier_uniform_, constant_

import copy, math


def clones(module: nn.Module = None, N: int = None):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int = None):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor = None):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs: int = None, n_outputs: int = None, kernel_size: int = None, 
        stride: int = None, dilation: int = None, padding: int = None, dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor = None):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int = None, num_channels: int = None, kernel_size: int = 2, dropout: float = 0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_channels = num_channels
        num_levels = len(num_channels)
        self.out_channels = None
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size, dropout=dropout
                )]
            
            self.out_channels = out_channels

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor = None):
        return self.network(x.transpose(1, 2)).transpose(1, 2) * math.sqrt(self.out_channels)