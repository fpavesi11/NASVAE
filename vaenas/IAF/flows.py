import math

import torch 
import torch.nn as nn
from torch.nn.init import xavier_normal_ as xavier_normal
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# STOLEN FROM https://github.com/kefirski/bdir_vae/tree/master
# It uses MADE approach, which is standard for general input


class AutoregressiveLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True, ):
        super(AutoregressiveLinear, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = xavier_normal(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight.tril(-1))

        output = input @ self.weight.tril(-1)
        if self.bias is not None:
            output += self.bias
        return output
    


###
# HIGHWAY AS IMPLEMENTED IN GERMAIN 2018 PAPER (MADE) WITH THE SUGGESTION
# FROM IAF PAPER OF ADDING HIDDEN STATE
###

class HighwayStandard(nn.Module):
    def __init__(self, size, bias=True):
        
        super(HighwayStandard, self).__init__()

        self.size=size

        self.weight_x = Parameter(torch.Tensor(size, size))
        
        self.weight_h = Parameter(torch.Tensor(size, size))

        if bias:
            self.bias = Parameter(torch.Tensor(size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.size)

        self.weight_x = xavier_normal(self.weight_x)
        self.weight_h = xavier_normal(self.weight_h)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, h):
        """
        h(x) = (W_x * M) @ x + W_h @ h + b    
        """
        out = x @ self.weight_x.tril(-1) + h @ self.weight_h
        if self.bias is not None:
            out += self.bias
        return out




####
# HIGHWAY AS IMPLEMENTED IN THIS REPOSITORY
###    
class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.utils.parametrizations.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.utils.parametrizations.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.utils.parametrizations.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ f(G(x)) + (1 - σ(x)) ⨀ Q(x) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = nn.Sigmoid()(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x