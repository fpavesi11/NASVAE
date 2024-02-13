from torch import nn 
import torch
from cfg_nas.RecurrentTranslator import PickLast, KeepHidden


"""
LINEAR DECODER STANDARD
"""

class LinearDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dimension, max_length):
        super(LinearDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.input_size = input_size
        self.max_length = max_length
        self.inner_structure = nn.Sequential(nn.Linear(in_features=latent_dimension,
                                                       out_features=512),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(in_features=512,
                                                       out_features=512),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(in_features=512,
                                                          out_features=512),
                                            nn.LeakyReLU(0.2))
        self.predict_h = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=512),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(in_features=512,
                                                 out_features=hidden_size))
        self.predict_c = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=512),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(in_features=512,
                                                 out_features=hidden_size))
        self.predict_x_0 = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=512),
                                         nn.LeakyReLU(0.2),
                                         nn.Linear(in_features=512,
                                                   out_features=input_size))
        self.autoregressiveDecoder = AutoRegressiveDecoder(input_size=input_size, 
                                                           hidden_size=hidden_size, 
                                                           max_length=max_length)
        

    
    def forward(self, x):
        x = self.inner_structure(x)
        
        h = self.predict_h(x)
        c = self.predict_c(x)
        x_0 = self.predict_x_0(x)
        
        layer_type, features = torch.split(x_0, [x_0.size(-1)-1, 1], dim=-1)
        layer_type = nn.Softmax(dim=-1)(layer_type)
        features = nn.Sigmoid()(features)
        x_0 = torch.cat([layer_type, features], dim=-1)
        
        return self.autoregressiveDecoder(x_0, h, c)
        
    
    
class AutoRegressiveDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_length):
        super(AutoRegressiveDecoder, self).__init__()
        self.max_length = max_length
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True)
        self.predictor = nn.Linear(hidden_size, input_size)
        
        
    def decode_step(self, x, h, c):
        out, _ = self.rnn(x, (h, c))
        return out[:,-1,:]
        
    def forward(self, x, h, c):
        x_out = x.unsqueeze(1)
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        for i in range(self.max_length):
            new_x_t = self.decode_step(x_out, h, c)
            new_x_t = self.predictor(new_x_t)
            layer_type, features = torch.split(new_x_t, [new_x_t.size(-1)-1, 1], dim=-1)
            layer_type = nn.Softmax(dim=-1)(layer_type)
            features = nn.Sigmoid()(features)
            new_x_t = torch.cat([layer_type, features], dim=-1)
            new_x_t = new_x_t.unsqueeze(1)
            x_out = torch.cat([x_out, new_x_t], dim=1)
        return x_out[:,1:,:] #remove first output which is sequence start
    


"""
LINEAR DECODER TEACHER FORCING
"""

class LinearDecoderTF(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dimension, max_length, use_predict=False):
        super(LinearDecoderTF, self).__init__()
        self.latent_dimension = latent_dimension
        self.input_size = input_size
        self.max_length = max_length
        self.inner_structure = nn.Sequential(nn.Linear(in_features=latent_dimension,
                                                       out_features=512),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(in_features=512,
                                                       out_features=512),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(in_features=512,
                                                          out_features=512),
                                            nn.LeakyReLU(0.2))
        self.predict_h = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=512),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(in_features=512,
                                                 out_features=hidden_size))
        self.predict_c = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=512),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(in_features=512,
                                                 out_features=hidden_size))
        self.predict_x_0 = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=512),
                                         nn.LeakyReLU(0.2),
                                         nn.Linear(in_features=512,
                                                   out_features=input_size))
        self.autoregressiveDecoder = AutoRegressiveDecoderTF(input_size=input_size, 
                                                           hidden_size=hidden_size, 
                                                           max_length=max_length,
                                                           use_predict=use_predict)
        

    
    def forward(self, x, x_true=None):
        x = self.inner_structure(x)
        
        h = self.predict_h(x)
        c = self.predict_c(x)
        x_0 = self.predict_x_0(x)
        
        layer_type, features = torch.split(x_0, [x_0.size(-1)-1, 1], dim=-1)
        layer_type = nn.Softmax(dim=-1)(layer_type)
        features = nn.Sigmoid()(features)
        x_0 = torch.cat([layer_type, features], dim=-1)
        
        return self.autoregressiveDecoder(x_0, h, c, x_true)

    
class AutoRegressiveDecoderTF(nn.Module):
    def __init__(self, input_size, hidden_size, max_length, use_predict=False):
        super(AutoRegressiveDecoderTF, self).__init__()
        self.input_size = input_size
        self.max_length = max_length
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True)
        self.predictor = nn.Linear(hidden_size, input_size)
        self.use_predict = use_predict
        
        
    def decode_step(self, x, h, c):
        out, _ = self.rnn(x, (h, c))
        return out[:,-1,:]
    
    def decode_no_tf(self, x, h, c):
        x_out = x.unsqueeze(1)
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        for i in range(self.max_length):
            new_x_t = self.decode_step(x_out, h, c)
            new_x_t = self.predictor(new_x_t)
            layer_type, features = torch.split(new_x_t, [new_x_t.size(-1)-1, 1], dim=-1)
            layer_type = nn.Softmax(dim=-1)(layer_type)
            features = nn.Sigmoid()(features)
            new_x_t = torch.cat([layer_type, features], dim=-1)
            new_x_t = new_x_t.unsqueeze(1)
            x_out = torch.cat([x_out, new_x_t], dim=1)
        return x_out[:,1:,:] #remove first output which is sequence start
    
    def predict(self, x, h, c):
        x_out = x.unsqueeze(1)
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        for i in range(self.max_length):
            new_x_t = self.decode_step(x_out, h, c)
            new_x_t = self.predictor(new_x_t)
            layer_type, features = torch.split(new_x_t, [new_x_t.size(-1)-1, 1], dim=-1)
            layer_type = nn.Softmax(dim=-1)(layer_type)
            argmax = torch.argmax(layer_type, dim=-1)
            num_classes = layer_type.size(-1)
            layer_type = torch.eye(num_classes)[argmax]
            features = nn.Sigmoid()(features)
            new_x_t = torch.cat([layer_type, features], dim=-1)
            new_x_t = new_x_t.unsqueeze(1)
            x_out = torch.cat([x_out, new_x_t], dim=1)
        return x_out[:,1:,:] #remove first output which is sequence start
    
    def decode_tf(self, x, x_true, h, c):
        # we add the predicted x as first element of sequence and remove last element of sequence
        x_in = torch.cat([x.unsqueeze(1), x_true], dim=1)[:,:-1,:] 
        # teacher forcing rnn
        x_out, _ = self.rnn(x_in, (h.unsqueeze(0), c.unsqueeze(0)))
        # element-wise linear 
        new_x_out = torch.empty((x_out.size(0), x_out.size(1), self.input_size))
        for i in range(x_out.size(1)):
            new_x_out[:,i,:] = self.predictor(x_out[:,i,:])
        # apply activations
        layer_type, features = torch.split(new_x_out, [new_x_out.size(-1)-1, 1], dim=-1)
        layer_type = nn.Softmax(dim=-1)(layer_type)
        features = nn.Sigmoid()(features)
        new_x_out = torch.cat([layer_type, features], dim=-1)
        return new_x_out
        
    def forward(self, x, h, c, x_true=None):
        if x_true is not None:
            x_out = self.decode_tf(x, x_true, h, c)
        elif x_true is None and self.use_predict:
            x_out = self.predict(x, h, c)
        else:
            x_out = self.decode_no_tf(x, h, c)
        return x_out
    
"""
LINEAR DECODER VERSION 2
"""

class LinearDecoderV2(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dimension, max_length):
        super(LinearDecoderV2, self).__init__()
        self.latent_dimension = latent_dimension
        self.input_size = input_size
        self.max_length = max_length
        self.inner_structure = nn.Sequential(nn.Linear(in_features=latent_dimension,
                                                       out_features=512),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(in_features=512,
                                                       out_features=512),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(in_features=512,
                                                          out_features=512),
                                            nn.LeakyReLU(0.2))
        self.predict_h = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=512),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(in_features=512,
                                                 out_features=hidden_size))
        self.predict_c = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=512),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(in_features=512,
                                                 out_features=hidden_size))
        self.predict_x_0 = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=512),
                                         nn.LeakyReLU(0.2),
                                         nn.Linear(in_features=512,
                                                   out_features=hidden_size))
        self.autoregressiveDecoder = AutoRegressiveDecoderV2(input_size=input_size, 
                                                           hidden_size=hidden_size, 
                                                           max_length=max_length)
        

    
    def forward(self, x):
        x = self.inner_structure(x)
        
        h = self.predict_h(x)
        c = self.predict_c(x)
        x_0 = self.predict_x_0(x)
        
        return self.autoregressiveDecoder(x_0, h, c)
    
class AutoRegressiveDecoderV2(nn.Module):
    def __init__(self, input_size, hidden_size, max_length):
        super(AutoRegressiveDecoderV2, self).__init__()
        self.input_size = input_size
        self.max_length = max_length
        self.rnn = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True)
        self.predictor = nn.LSTM(input_size=hidden_size,
                                 hidden_size=input_size,
                                 num_layers=1,
                                 batch_first=True)
        
        
    def decode_step(self, x, h, c):
        out, _ = self.rnn(x, (h, c))
        return out[:,-1,:]
    
    def forward(self, x, h, c):
        x_out = x.unsqueeze(1)
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        for i in range(self.max_length):
            new_x_t = self.decode_step(x_out, h, c)
            new_x_t = new_x_t.unsqueeze(1)
            x_out = torch.cat([x_out, new_x_t], dim=1)
            
        new_x_out, _ = self.predictor(x_out[:,1:,:])#remove first output which is sequence start
            
        layer_type, features = torch.split(new_x_out, [new_x_out.size(-1)-1, 1], dim=-1)
        layer_type = nn.Softmax(dim=-1)(layer_type)
        features = nn.Sigmoid()(features)
        new_x_out = torch.cat([layer_type, features], dim=-1)
        return new_x_out
    
