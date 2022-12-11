import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Linear + Norm + ReLU Module
    """
    def __init__(self, n_in, n_out, norm='BN', act=True):
        """
        :param n_in: input dimension
        :param n_out: output dimension
        :param norm: norm name, e.g. 'BN' for BatchNorm, 'GN' for GroupNorm
        :param act: indicator for whether to invoke activation layer
        """
        super().__init__()

        self.linear = nn.Linear(n_in, n_out, bias=False)
        if norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.act = act


    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act: # if invoke activation
            out = self.relu(out)
        return out


class Conv1d(nn.Module):
    """
    Conv1d + Norm + ReLU Module
    """
    def __init__(self, n_in, n_out, kernel_size=3, padding=1, stride=1, norm='BN', act=True):
        """
        :param n_in: input channel
        :param n_out: output channel
        :param kernel_size: 
        :param stride:
        :param norm: norm name, e.g. 'BN' for BatchNorm, 'GN' for GroupNorm
        :param act: indicator for whether to invoke activation layer
        """
        super().__init__()

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, \
                              padding=padding, stride=stride, bias=False)
        if norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.act = act


    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act: # if invoke activation
            out = self.relu(out)
        return out
  

class LinearRes(nn.Module):
    """
    Linear Residual Block, add skip connection between input and output of two linear layers
    """
    def __init__(self, n_in, n_out, norm='BN'):
        """
        :param n_in: input dimension
        :param n_out: output dimension
        :param norm: norm name, e.g. 'BN' for BatchNorm, 'GN' for GroupNorm
        """
        super().__init__()

        self.linear1 = nn.Linear(n_in, n_out, bias = False)
        self.linear2 = nn.Linear(n_out, n_out, bias = False)
        self.relu = nn.ReLU(inplace = True)

        if norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)

        self.transform = None
        if n_in != n_out:
            self.transform = nn.Sequential(
                nn.Linear(n_in, n_out, bias = False),
                nn.BatchNorm1d(n_out))
        

    def forward(self, x):
        output = self.relu(self.norm1(self.linear1(x)))
        output = self.norm2(self.linear2(output))

        # skip connection between input and linear layer output
        if self.transform:
            output = self.transform(x)

        output = self.relu(output)
        return output

