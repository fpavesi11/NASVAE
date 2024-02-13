import torch
from torch.nn.modules.loss import _Loss
from torch import nn


class NetworkEncoder: 
    def __init__(self, grammar, max_length, dtype=torch.float32):
        self.grammar = grammar
        self.dtype = dtype
        self.max_length = max_length

    def encode_linear(self, x):
        out_encoding = []
        for module in x:
            if module[0] == '<start>':
                layer_encoding = torch.tensor([1,0,0,0,0], dtype=self.dtype)
            elif module[0] == 'linear':
                layer_encoding = torch.tensor([0,1,0,0,0], dtype=self.dtype)
            elif module[0] == 'dropout':
                layer_encoding = torch.tensor([0,0,1,0,0], dtype=self.dtype)
            elif module[0] == '<end>':
                layer_encoding = torch.tensor([0,0,0,1,0], dtype=self.dtype)
            else:
                raise ValueError('Layer {layer} does no belong to Linear grammar'.format(layer=str(module[0])))

            features_encoding = torch.tensor(module[1], dtype=self.dtype).unsqueeze(0)
            
            out_encoding.append(torch.cat([layer_encoding, features_encoding]).unsqueeze(0))
        
        if len(out_encoding) < self.max_length:
            for j in range(self.max_length - len(out_encoding)):
                out_encoding.append(torch.tensor([0,0,0,0,1, 0], dtype=self.dtype).unsqueeze(0)) #last zero is features
        elif len(out_encoding) > self.max_length:
            raise ValueError('The network exceeds max_length')
        out_encoding = torch.cat(out_encoding, dim=0).unsqueeze(0)
        return out_encoding
    
    def encode(self, x):
        if self.grammar == 'linear':
            encoder = self.encode_linear
        elif self.grammar == 'convolutional':
            raise NotImplementedError('Not yet implemented')
        elif self.grammar == 'recurrent':
            raise NotImplementedError('Not yet implemented')
        else:
            raise ValueError('Unknown "{grammar}" grammar'.format(grammar=self.grammar))
        
        encoded_networks = []
        for network in x:
            encoded_networks.append(encoder(network))
        encoded_networks = torch.cat(encoded_networks, dim=0)
        return encoded_networks
    
    
"""
CUSTOM RECOGNITION LOSS FOR LINEAR 
"""

class CustomReconLosslinear(_Loss):
    def __init__(self, reduction='sum'):
        super(CustomReconLosslinear, self).__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.celoss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, x_hat, x):
        features_loss = self.mse_loss(x_hat[:,:,-1], x[:,:,-1]).sum(-1, keepdim=True)
        category_loss = self.celoss(x_hat[:,:,:-1], x[:,:,:-1]).sum(-1, keepdim=True)
        return features_loss, category_loss
    
    
"""
LOG NORMALIZER: Normalizes last colum in logarithmic scale + 1
"""

def log_norm(x):
    return torch.log(x[:,:,-1] + 1)

def exp_norm(x):
    return torch.exp(x[:,:,-1]) - 1


"""
MAX VAL NORMALIZER: Normalizes all features between 0 and 1 (except dropout)
"""

def max_norm(x, max_val=1028):
    eligible_nodes = (x[:,:,2] == 0).unsqueeze(-1) #nodes not dropout
    normalized_nodes = ((x * eligible_nodes)[:,:,-1] / max_val).unsqueeze(-1)
    dropout_nodes = ((x * (eligible_nodes == 0))[:,:,-1]).unsqueeze(-1)
    out_nodes = normalized_nodes + dropout_nodes
    return torch.cat([x[:,:,:-1], out_nodes], dim=-1)


def inv_max_norm(x, max_val=1028):
    eligible_nodes = (x[:,:,2] == 0).unsqueeze(-1) #nodes not dropout
    normalized_nodes = ((x * eligible_nodes)[:,:,-1] * max_val).unsqueeze(-1)
    dropout_nodes = ((x * (eligible_nodes == 0))[:,:,-1]).unsqueeze(-1)
    out_nodes = normalized_nodes + dropout_nodes
    return torch.cat([x[:,:,:-1], out_nodes], dim=-1) 

"""
FIXES NETWORK PREDICTION: makes predicted network readable
"""

def fix_network(x):
    # layer predicted
    predicted_layer = torch.argmax(x[:,:,:-1], dim=-1)
    num_classes = x[:,:,:-1].size(-1)
    one_hot_encoded_predictions = torch.eye(num_classes)[predicted_layer]
    int_out = torch.cat([one_hot_encoded_predictions, x[:,:,-1].unsqueeze(-1)], dim=-1)
    
    # features predicted
    out = inv_max_norm(int_out)
    
    return out